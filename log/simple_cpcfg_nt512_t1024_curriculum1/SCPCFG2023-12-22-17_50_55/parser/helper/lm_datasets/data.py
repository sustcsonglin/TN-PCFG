

from __future__ import division

import math
import random

import logging

import torch as th
from torch.utils.data import DataLoader
import numpy as np

from torchtext import data
import io

from torchtext.data.utils import RandomShuffler
from torchtext.data.dataset import Dataset
from torchtext.data.iterator import BPTTIterator

logger = logging.getLogger(__name__)


class Batch(object):
    """Defines a batch of examples along with its Fields.

    Attributes:
        batch_size: Number of examples in the batch.
        dataset: A reference to the dataset object the examples come from
            (which itself contains the dataset's Field objects).
        train: Deprecated: this attribute is left for backwards compatibility,
            however it is UNUSED as of the merger with pytorch 0.4.
        input_fields: The names of the fields that are used as input for the model
        target_fields: The names of the fields that are used as targets during
                       model training

    Also stores the Variable for each column in the batch as an attribute.
    """

    def __init__(self, data=None, dataset=None, device=None):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.fields = dataset.fields.keys()  # copy field names
            self.input_fields = [k for k, v in dataset.fields.items() if
                                 v is not None and not v.is_target]
            self.target_fields = [k for k, v in dataset.fields.items() if
                                  v is not None and v.is_target]

            for (name, field) in dataset.fields.items():
                if field is not None:
                    batch = [getattr(x, name) for x in data]
                    setattr(self, name, field.process(batch, device=device))
            self.idxs = [x.idx for x in data]

    @classmethod
    def fromvars(cls, dataset, batch_size, train=None, **kwargs):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        batch.batch_size = batch_size
        batch.dataset = dataset
        batch.fields = dataset.fields.keys()
        for k, v in kwargs.items():
            setattr(batch, k, v)
        return batch

    def __repr__(self):
        return str(self)

    def __str__(self):
        if not self.__dict__:
            return 'Empty {} instance'.format(th.typename(self))

        fields_to_index = filter(lambda field: field is not None, self.fields)
        var_strs = '\n'.join(['\t[.' + name + ']' + ":" + _short_str(getattr(self, name))
                              for name in fields_to_index if hasattr(self, name)])

        data_str = (' from {}'.format(self.dataset.name.upper())
                    if hasattr(self.dataset, 'name')
                    and isinstance(self.dataset.name, str) else '')

        strt = '[{} of size {}{}]\n{}'.format(th.typename(self),
                                              self.batch_size, data_str, var_strs)
        return '\n' + strt

    def __len__(self):
        return self.batch_size

    def _get_field_values(self, fields):
        if len(fields) == 0:
            return None
        elif len(fields) == 1:
            return getattr(self, fields[0])
        else:
            return tuple(getattr(self, f) for f in fields)

    def __iter__(self):
        yield self._get_field_values(self.input_fields)
        yield self._get_field_values(self.target_fields)


def _short_str(tensor):
    # unwrap variable to tensor
    if not th.is_tensor(tensor):
        # (1) unpack variable
        if hasattr(tensor, 'data'):
            tensor = getattr(tensor, 'data')
        # (2) handle include_lengths
        elif isinstance(tensor, tuple):
            return str(tuple(_short_str(t) for t in tensor))
        # (3) fallback to default str
        else:
            return str(tensor)

    # copied from torch _tensor_str
    size_str = 'x'.join(str(size) for size in tensor.size())
    device_str = '' if not tensor.is_cuda else \
        ' (GPU {})'.format(tensor.get_device())
    strt = '[{} of size {}{}]'.format(th.typename(tensor),
                                      size_str, device_str)
    return strt


class Iterator(object):
    """Defines an iterator that loads batches of data from a Dataset.

    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
        batch_size_fn: Function of three arguments (new example to add, current
            count of examples in the batch, and current effective batch size)
            that returns the new effective batch size resulting from adding
            that example to a batch. This is useful for dynamic batching, where
            this function would add to the current effective batch size the
            number of tokens in the new example.
        sort_key: A key to use for sorting examples in order to batch together
            examples with similar lengths and minimize padding. The sort_key
            provided to the Iterator constructor overrides the sort_key
            attribute of the Dataset, or defers to it if None.
        train: Whether the iterator represents a train set.
        repeat: Whether to repeat the iterator for multiple epochs. Default: False.
        shuffle: Whether to shuffle examples between epochs.
        sort: Whether to sort examples according to self.sort_key.
            Note that shuffle and sort default to train and (not train).
        sort_within_batch: Whether to sort (in descending order according to
            self.sort_key) within each batch. If None, defaults to self.sort.
            If self.sort is True and this is False, the batch is left in the
            original (ascending) sorted order.
        device (str or `torch.device`): A string or instance of `torch.device`
            specifying which device the Variables are going to be created on.
            If left as default, the tensors will be created on cpu. Default: None.
    """

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None, train=True,
                 repeat=False, shuffle=None, sort=None,
                 sort_within_batch=None):
        self.batch_size, self.train, self.dataset = batch_size, train, dataset
        self.batch_size_fn = batch_size_fn
        self.iterations = 0
        self.repeat = repeat
        self.shuffle = train if shuffle is None else shuffle
        self.sort = not train if sort is None else sort

        if sort_within_batch is None:
            self.sort_within_batch = self.sort
        else:
            self.sort_within_batch = sort_within_batch
        if sort_key is None:
            self.sort_key = dataset.sort_key
        else:
            self.sort_key = sort_key

        if isinstance(device, int):
            logger.warning("The `device` argument should be set by using `torch.device`"
                           + " or passing a string as an argument. This behavior will be"
                           + " deprecated soon and currently defaults to cpu.")
            device = None
        self.device = device
        self.random_shuffler = RandomShuffler()

        # For state loading/saving only
        self._iterations_this_epoch = 0
        self._random_state_this_epoch = None
        self._restored_from_state = False

    @classmethod
    def splits(cls, datasets, batch_sizes=None, **kwargs):
        """Create Iterator objects for multiple splits of a dataset.

        Arguments:
            datasets: Tuple of Dataset objects corresponding to the splits. The
                first such object should be the train set.
            batch_sizes: Tuple of batch sizes to use for the different splits,
                or None to use the same batch_size for all splits.
            Remaining keyword arguments: Passed to the constructor of the
                iterator class being used.
        """
        if batch_sizes is None:
            batch_sizes = [kwargs.pop('batch_size')] * len(datasets)
        ret = []
        for i in range(len(datasets)):
            train = i == 0
            ret.append(cls(
                datasets[i], batch_size=batch_sizes[i], train=train, **kwargs))
        return tuple(ret)

    def data(self):
        """Return the examples in the dataset in order, sorted, or shuffled."""
        if self.sort:
            #xs = sorted(self.dataset, key=self.sort_key)
            xs = self.dataset
        elif self.shuffle:
            xs = [self.dataset[i] for i in self.random_shuffler(range(len(self.dataset)))]
        else:
            xs = self.dataset
        return xs

    def init_epoch(self):
        """Set up the batch generator for a new epoch."""

        if self._restored_from_state:
            self.random_shuffler.random_state = self._random_state_this_epoch
        else:
            self._random_state_this_epoch = self.random_shuffler.random_state

        self.create_batches()

        if self._restored_from_state:
            self._restored_from_state = False
        else:
            self._iterations_this_epoch = 0

        if not self.repeat:
            self.iterations = 0

    def create_batches(self):
        self.batches = batch(self.data(), self.batch_size, self.batch_size_fn)

    @property
    def epoch(self):
        return math.floor(self.iterations / len(self))

    def __len__(self):
        if self.batch_size_fn is not None:
            raise NotImplementedError
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                yield Batch(minibatch, self.dataset, self.device)
            if not self.repeat:
                return

    def state_dict(self):
        return {
            "iterations": self.iterations,
            "iterations_this_epoch": self._iterations_this_epoch,
            "random_state_this_epoch": self._random_state_this_epoch}

    def load_state_dict(self, state_dict):
        self.iterations = state_dict["iterations"]
        self._iterations_this_epoch = state_dict["iterations_this_epoch"]
        self._random_state_this_epoch = state_dict["random_state_this_epoch"]
        self._restored_from_state = True


class BPTTIterator(Iterator):
    """Defines an iterator for language modeling tasks that use BPTT.

    Provides contiguous streams of examples together with targets that are
    one timestep further forward, for language modeling training with
    backpropagation through time (BPTT). Expects a Dataset with a single
    example and a single field called 'text' and produces Batches with text and
    target attributes.

    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
        bptt_len: Length of sequences for backpropagation through time.
        sort_key: A key to use for sorting examples in order to batch together
            examples with similar lengths and minimize padding. The sort_key
            provided to the Iterator constructor overrides the sort_key
            attribute of the Dataset, or defers to it if None.
        train: Whether the iterator represents a train set.
        repeat: Whether to repeat the iterator for multiple epochs. Default: False.
        shuffle: Whether to shuffle examples between epochs.
        sort: Whether to sort examples according to self.sort_key.
            Note that shuffle and sort default to train and (not train).
        device (str or torch.device): A string or instance of `torch.device`
            specifying which device the Variables are going to be created on.
            If left as default, the tensors will be created on cpu. Default: None.
    """

    def __init__(self, dataset, batch_size, bptt_len, **kwargs):
        self.bptt_len = bptt_len
        super(BPTTIterator, self).__init__(dataset, batch_size, **kwargs)

    def __len__(self):
        return math.ceil((len(self.dataset[0].text) / self.batch_size - 1)
                         / self.bptt_len)

    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None
        text = text + ([TEXT.pad_token] * int(math.ceil(len(text) / self.batch_size)
                                              * self.batch_size - len(text)))
        data = TEXT.numericalize(
            [text], device=self.device)
        data = data.view(self.batch_size, -1).t().contiguous()
        dataset = Dataset(examples=self.dataset.examples, fields=[
            ('text', TEXT), ('target', TEXT)])
        while True:
            for i in range(0, len(self) * self.bptt_len, self.bptt_len):
                self.iterations += 1
                seq_len = min(self.bptt_len, len(data) - i - 1)
                batch_text = data[i:i + seq_len]
                batch_target = data[i + 1:i + 1 + seq_len]
                if TEXT.batch_first:
                    batch_text = batch_text.t().contiguous()
                    batch_target = batch_target.t().contiguous()
                yield Batch.fromvars(
                    dataset, self.batch_size,
                    text=batch_text,
                    target=batch_target)
            if not self.repeat:
                return


class BucketIterator(Iterator):
    """Defines an iterator that batches examples of similar lengths together.

    Minimizes amount of padding needed while producing freshly shuffled
    batches for each new epoch. See pool for the bucketing procedure used.
    """

    def create_batches(self):
        if self.sort:
            self.batches = batch(self.data(), self.batch_size,
                                 self.batch_size_fn)
        else:
            self.batches = pool(self.data(), self.batch_size,
                                self.sort_key, self.batch_size_fn,
                                random_shuffler=self.random_shuffler,
                                shuffle=self.shuffle,
                                sort_within_batch=self.sort_within_batch)
        # print(self.batches)

    def __len__(self):
        # coarse batch (actually wrong)
        return math.ceil(sum(len(x.text) for x in self.dataset) / self.batch_size)
        #if self.batch_size_fn is not None:
            #raise NotImplementedError
        #return math.ceil(len(self.dataset) / self.batch_size)



def batch(data, batch_size, batch_size_fn=None):
    """Yield elements from data in chunks of batch_size."""
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return max(len(new.text), sofar)
            #return count
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        count = len(minibatch)
        size_so_far = batch_size_fn(ex, count, size_so_far)
        if size_so_far * count == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far * count > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
    if minibatch:
        yield minibatch


def pool(data, batch_size, key, batch_size_fn=lambda new, count, sofar: count,
         random_shuffler=None, shuffle=False, sort_within_batch=False):
    """Sort within buckets, then batch, then shuffle batches.

    Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.
    """
    if random_shuffler is None:
        random_shuffler = random.shuffle
    for p in batch(data, batch_size * 100, batch_size_fn):
        p_batch = batch(sorted(p, key=key), batch_size, batch_size_fn) \
            if sort_within_batch \
            else batch(p, batch_size, batch_size_fn)
        if shuffle:
            for b in random_shuffler(list(p_batch)):
                yield b
        else:
            for b in list(p_batch):
                yield b

class BPTTIterator(Iterator):
    """Defines an iterator for language modeling tasks that use BPTT.
    Provides contiguous streams of examples together with targets that are
    one timestep further forward, for language modeling training with
    backpropagation through time (BPTT). Expects a Dataset with a single
    example and a single field called 'text' and produces Batches with text and
    target attributes.
    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
        bptt_len: Length of sequences for backpropagation through time.
        sort_key: A key to use for sorting examples in order to batch together
            examples with similar lengths and minimize padding. The sort_key
            provided to the Iterator constructor overrides the sort_key
            attribute of the Dataset, or defers to it if None.
        train: Whether the iterator represents a train set.
        repeat: Whether to repeat the iterator for multiple epochs. Default: False.
        shuffle: Whether to shuffle examples between epochs.
        sort: Whether to sort examples according to self.sort_key.
            Note that shuffle and sort default to train and (not train).
        device (str or torch.device): A string or instance of `torch.device`
            specifying which device the Variables are going to be created on.
            If left as default, the tensors will be created on cpu. Default: None.
    """

    def __init__(self, dataset, batch_size, bptt_len, **kwargs):
        self.bptt_len = bptt_len
        super(BPTTIterator, self).__init__(dataset, batch_size, **kwargs)

    def __len__(self):
        length = sum([len(x.text) for x in self.dataset])
        return math.ceil((length / self.batch_size - 1) / self.bptt_len)

    def __iter__(self):
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None
        # ^ not sure what this is for

        dataset = Dataset(
            examples=self.dataset.examples,
            fields=[
                ('text', TEXT), #('target', TEXT)
            ],
        )

        # double check this is deterministic during eval
        raw_data = self.data()
        # concatenate everything
        text = [word for ex in raw_data for word in ex.text]
        text = text + (
            [TEXT.pad_token]
            * int(
                math.ceil(len(text) / self.batch_size)
                * self.batch_size
                - len(text)
            )
        )
        data = TEXT.numericalize(
            [text], device=self.device)
        data = data.view(self.batch_size, -1).t()
        eos_token = TEXT.vocab["<eos>"]
        data = th.cat([
            th.empty(
                self.batch_size,
                device=data.device,
                dtype=data.dtype,
            )[None].fill_(eos_token),
            data,
        ], 0)

        while True:
            for i in range(1, len(self) * self.bptt_len, self.bptt_len):
                self.iterations += 1
                """
                seq_len = min(self.bptt_len, len(data) - i - 1)
                batch_text = data[i:i + seq_len]
                batch_target = data[i + 1:i + 1 + seq_len]
                if TEXT.batch_first:
                    batch_text = batch_text.t().contiguous()
                    batch_target = batch_target.t().contiguous()
                yield Batch.fromvars(
                    dataset, self.batch_size,
                    text=batch_text,
                    target=batch_target)
                """
                seq_len = min(self.bptt_len, len(data) - i)
                batch_text = data[i:i + seq_len]
                batch_textp1 = data[i-1:i + seq_len]
                if TEXT.batch_first:
                    batch_text = batch_text.t().contiguous()
                    batch_textp1 = batch_textp1.t().contiguous()
                yield Batch.fromvars(
                    dataset, self.batch_size,
                    text=batch_text,
                    textp1=batch_textp1,
                )
            if not self.repeat:
                return
