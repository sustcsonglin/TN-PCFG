import torchtext
from .lm_datasets import PennTreebank, WikiText2
from .data import BucketIterator, BPTTIterator
from torch.utils.data import Sampler
from collections import defaultdict
import os
import random

class LMDataModule():
	def __init__(self, hparams):
		super().__init__()

		self.hparams = hparams
		self.device = self.hparams.device
		self.setup()

	def prepare_data(self):
		pass

	def setup(self):
	
		TEXT = torchtext.data.Field(batch_first = True)
	
		dataset = self.hparams.data.dataset
		if dataset == "ptb":
			Dataset = PennTreebank
		elif dataset == "wikitext2":
			Dataset = WikiText2
		else:
			raise ValueError("Unsupported dataset")
		
		train_dataset, val_dataset, test_dataset = Dataset.splits(
										TEXT, newline_eos = True, debug = self.hparams.debug, 
          								root = self.hparams.data.root, train = self.hparams.data.train_file, 
                  						validation = self.hparams.data.val_file, test = self.hparams.data.test_file)
										
		TEXT.build_vocab(train_dataset)
		self.V = TEXT.vocab
		
		def batch_size_tokens(new, count, sofar):
			return max(len(new.text), sofar)
		def batch_size_sents(new, count, sofar):
			return count
			
		if self.hparams.iterator == 'bucket':
			train_iter, valid_iter, test_iter = BucketIterator.splits(
			(train_dataset, val_dataset, test_dataset),
			batch_sizes = [self.hparams.train.train_bsz, self.hparams.eval.eval_bsz, self.hparams.eval.eval_bsz],
			device = self.device,
			sort_key = lambda x: len(x.text),
			batch_size_fn = batch_size_tokens if self.hparams.bsz_fn == "tokens" else batch_size_sents,
		)
		
		elif self.hparams.iterator == "bptt":
			train_iter, valid_iter, test_iter = BPTTIterator.splits(
			(train_dataset, val_dataset, test_dataset),
			batch_sizes = [self.hparams.bsz, self.hparams.eval_bsz, self.hparams.eval_bsz],
			device = self.device,
			bptt_len = self.hparams.bptt,
			sort = False,
		)
		else:
			raise ValueError(f"Invalid iterator {self.hparams.iterator}")
			
		self.train_dataloader = train_iter
		self.val_dataloader = valid_iter
		self.test_dataloader = test_iter






