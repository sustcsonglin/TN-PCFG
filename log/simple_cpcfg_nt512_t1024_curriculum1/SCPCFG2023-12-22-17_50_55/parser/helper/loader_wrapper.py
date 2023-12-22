import torch

'''
Automatically put the tensor into the given device.
'''
class LoaderWrapper():
    def __init__(self, loader, device):
        self.loader = loader
        self.loader_iter = iter(loader)
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        batch_x, batch_y = next(self.loader_iter)
        for name, key in batch_x.items():
            if type(key) is torch.Tensor:
                batch_x[name] = key.to(self.device)
        for name, key in batch_y.items():
            if type(key) is torch.Tensor:
                batch_y[name] = key.to(self.device)
        return batch_x, batch_y

    def __getitem__(self, item):
        return self.loader[item]



    def __len__(self):
        return len(self.loader)


'''
Accelerate data fetching.
'''
class DataPrefetcher:
    # https://github.com/NVIDIA/apex/blob/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet/main_amp.py#L256
    def __init__(self, loader, device, init=False):
        self.loader = LoaderWrapper(loader,device)
        self.iter = None
        self.stream = torch.cuda.Stream()

        if init:
            self.iter = iter(self.loader)
            self.preload()

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = [i.cuda(non_blocking=True) if isinstance(i, torch.Tensor) else i for i in self.next_batch]

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch

    def __iter__(self):
        self.iter = iter(self.loader)
        self.preload()
        while True:
            batch = self.next()
            if batch is None:
                break
            yield batch

