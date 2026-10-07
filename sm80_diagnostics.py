"""Temporary per-run SM80+ summaries; no extra GPU timers or tensor copies."""
import collections
import importlib.metadata
import logging

import torch

_LOG = logging.getLogger('MiniMaxH3ActivationChunkStar7')


class RunDiagnostics:
    def __init__(self, configured):
        self.configured = configured
        self.run_number = 0
        self._reset()

    def _reset(self):
        self.calls = collections.Counter()
        self.steps = collections.Counter()
        self.seconds = collections.Counter()
        self.qkv = set()
        self.device = None
        self.finite_outputs = 0
        self.allocated = 0
        self.implementations = set()

    def observe(self, route, tensors, chunk, weight_format):
        self.calls[route] += 1
        self.device = tensors[0].device
        if len(self.qkv) < 8:
            self.qkv.add((route, tuple(tensors[0].shape), str(tensors[0].dtype),
                          tuple(tuple(t.stride()) for t in tensors), chunk, weight_format))
        self.allocated = max(self.allocated, torch.cuda.memory_allocated(self.device))

    def step(self, route, seconds, device=None):
        self.steps[route] += 1
        self.seconds[route] += seconds
        if device is not None:
            self.device = device

    def producer(self, device, shape, dtype, chunk, weight_format):
        self.calls['Sol-official-producer'] += 1
        self.device = device
        if len(self.qkv) < 8:
            self.qkv.add(('Sol-official-producer', shape, str(dtype), (), chunk, weight_format))
        self.allocated = max(self.allocated, torch.cuda.memory_allocated(device))

    def finish(self):
        if not self.calls and not self.steps:
            self._reset()
            return
        self.run_number += 1
        device = self.device
        capability = torch.cuda.get_device_capability(device)
        try:
            triton_version = importlib.metadata.version('triton-windows')
        except importlib.metadata.PackageNotFoundError:
            try:
                triton_version = importlib.metadata.version('triton')
            except importlib.metadata.PackageNotFoundError:
                triton_version = 'unavailable'
        _LOG.info('[Star7 H3 SM80+] run=%d | revision=strided-veda-v1 | gpu=%s SM%d%d | torch=%s cuda=%s triton=%s | configured=%s implementations=%s | attention_calls=%s | timed_model_calls=%s model_seconds=%s | finite_outputs=%d | observed_allocated=%.2fGiB',
                  self.run_number, torch.cuda.get_device_name(device), *capability,
                  torch.__version__, torch.version.cuda, triton_version, self.configured, sorted(self.implementations),
                  dict(self.calls), dict(self.steps), {k: round(v, 3) for k, v in self.seconds.items()},
                  self.finite_outputs, self.allocated / 2**30)
        _LOG.info('[Star7 H3 SM80+] run=%d | QKV=(route, shape, dtype, Q/K/V strides, chunk, weight_format) %s',
                  self.run_number, sorted(self.qkv))
        self._reset()
