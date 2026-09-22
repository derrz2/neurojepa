import torch
import torch.distributed as dist

import time
import datetime
import logging

logger = logging.getLogger("neurojepa")

class MetricLogger:
    """Metric logger for training"""
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].update(v)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def log_every(self, iterable, print_freq, header=None, enable_log=True):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        log_msg = self.delimiter.join(log_msg)
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if enable_log and (i % print_freq == 0 or i == len(iterable) - 1):
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if dist.is_initialized():
                    if dist.get_rank() == 0:
                         logger.info(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time)))
                else:
                    logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if enable_log and (not dist.is_initialized() or dist.get_rank() == 0):
            logger.info(f'{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)')


class SmoothedValue:
    """Track a series of values and provide access to smoothed values"""
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = []
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.window_size = window_size

    def update(self, value, n=1):
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            self.deque.pop(0)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """Synchronize across all processes"""
        if not dist.is_available() or not dist.is_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = sorted(self.deque)
        n = len(d)
        if n == 0:
            return 0
        if n % 2 == 0:
            return (d[n // 2 - 1] + d[n // 2]) / 2
        return d[n // 2]

    @property
    def avg(self):
        if len(self.deque) == 0:
            return 0
        return sum(self.deque) / len(self.deque)

    @property
    def global_avg(self):
        if self.count == 0:
            return 0
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=max(self.deque) if len(self.deque) > 0 else 0,
            value=self.deque[-1] if len(self.deque) > 0 else 0
        )
