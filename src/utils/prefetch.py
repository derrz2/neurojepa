import torch

from .utils import move_to_device


def _record_stream_recursive(batch, stream):
    if torch.is_tensor(batch):
        batch.record_stream(stream)
    elif isinstance(batch, list):
        for item in batch:
            _record_stream_recursive(item, stream)
    elif isinstance(batch, tuple):
        for item in batch:
            _record_stream_recursive(item, stream)
    elif isinstance(batch, dict):
        for value in batch.values():
            _record_stream_recursive(value, stream)


class CUDAPrefetcher:
    """Prefetch CPU batch to CUDA stream for H2D/compute overlap."""

    def __init__(self, loader, device, non_blocking=True):
        self.loader = loader
        self.device = device
        self.non_blocking = non_blocking
        self.stream = torch.cuda.Stream(device=device)
        self.it = None
        self.next_batch = None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.it = iter(self.loader)
        self.next_batch = None
        self._preload()
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration

        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_stream(self.stream)

        batch = self.next_batch
        _record_stream_recursive(batch, current_stream)
        self._preload()
        return batch

    def _preload(self):
        try:
            batch = next(self.it)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = move_to_device(
                batch,
                self.device,
                non_blocking=self.non_blocking,
            )
