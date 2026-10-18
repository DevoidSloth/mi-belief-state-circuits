"""A small hook system, deliberately API-compatible with TransformerLens.

Hook names follow the TransformerLens convention (``blocks.0.attn.hook_z``,
``blocks.1.hook_resid_post``, ...) so that anyone who knows that library can read
this code, but the implementation is ~150 lines and has no external dependency,
which matters because every patching experiment in this repo runs through it.

Two operations cover everything downstream:

* **read** - :meth:`HookedRoot.run_with_cache` stashes named activations.
* **write** - :meth:`HookedRoot.hooks` installs functions that may *return a new
  tensor*, replacing the activation.  Patching, ablation and steering are all
  this one operation with different replacement values.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager

import torch
from torch import nn

HookFn = Callable[[torch.Tensor, str], torch.Tensor | None]


class HookPoint(nn.Module):
    """Identity module that exposes a named interception point."""

    def __init__(self) -> None:
        super().__init__()
        self.name: str = ""
        self._fns: list[HookFn] = []

    def add(self, fn: HookFn) -> None:
        self._fns.append(fn)

    def clear(self) -> None:
        self._fns.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fn in self._fns:
            out = fn(x, self.name)
            if out is not None:
                x = out
        return x


class HookedRoot(nn.Module):
    """Mixin giving a module named hook points, caching and temporary hooks."""

    def setup_hooks(self) -> None:
        """Assign every :class:`HookPoint` its dotted module path. Call after __init__."""
        self.hook_dict: dict[str, HookPoint] = {}
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.name = name
                self.hook_dict[name] = module

    def hook_names(self) -> list[str]:
        return list(self.hook_dict)

    def reset_hooks(self) -> None:
        for hp in self.hook_dict.values():
            hp.clear()

    def _resolve(self, spec: str | Iterable[str]) -> list[str]:
        if isinstance(spec, str):
            if spec in self.hook_dict:
                return [spec]
            matched = [n for n in self.hook_dict if spec in n]
            if not matched:
                raise KeyError(f"no hook matching {spec!r}; have {sorted(self.hook_dict)[:8]}...")
            return matched
        return [n for s in spec for n in self._resolve(s)]

    @contextmanager
    def hooks(self, fwd_hooks: Sequence[tuple[str, HookFn]] = ()) -> Iterator[HookedRoot]:
        """Install hooks for the duration of the block, then remove exactly those."""
        installed: list[tuple[HookPoint, HookFn]] = []
        try:
            for spec, fn in fwd_hooks:
                for name in self._resolve(spec):
                    hp = self.hook_dict[name]
                    hp.add(fn)
                    installed.append((hp, fn))
            yield self
        finally:
            for hp, fn in installed:
                if fn in hp._fns:
                    hp._fns.remove(fn)

    def run_with_hooks(self, *args, fwd_hooks: Sequence[tuple[str, HookFn]] = (), **kwargs):
        with self.hooks(fwd_hooks):
            return self(*args, **kwargs)

    def run_with_cache(
        self,
        *args,
        names: Iterable[str] | None = None,
        detach: bool = True,
        **kwargs,
    ):
        """Run the model and return ``(output, cache)``.

        ``names`` may be exact hook names or substrings; None caches everything.
        """
        wanted = set(self.hook_dict) if names is None else set(self._resolve(list(names)))
        cache: dict[str, torch.Tensor] = {}

        def make(name: str) -> HookFn:
            def fn(x: torch.Tensor, _n: str) -> None:
                cache[name] = x.detach() if detach else x
                return None

            return fn

        with self.hooks([(n, make(n)) for n in wanted]):
            out = self(*args, **kwargs)
        return out, cache


# --- hook factories used all over the analysis code -------------------------


def patch_hook(value: torch.Tensor, index: tuple | None = None) -> HookFn:
    """Replace an activation (or a slice of it) with ``value``."""

    def fn(x: torch.Tensor, _name: str) -> torch.Tensor:
        x = x.clone()
        if index is None:
            x[...] = value
        else:
            x[index] = value
        return x

    return fn


def add_hook(delta: torch.Tensor, index: tuple | None = None) -> HookFn:
    """Add ``delta`` to an activation (or a slice of it). Used for steering."""

    def fn(x: torch.Tensor, _name: str) -> torch.Tensor:
        x = x.clone()
        if index is None:
            x = x + delta
        else:
            x[index] = x[index] + delta
        return x

    return fn


def zero_hook(index: tuple | None = None) -> HookFn:
    """Zero-ablate an activation (or a slice of it)."""

    def fn(x: torch.Tensor, _name: str) -> torch.Tensor:
        x = x.clone()
        if index is None:
            return torch.zeros_like(x)
        x[index] = 0.0
        return x

    return fn


def mean_hook(mean: torch.Tensor, index: tuple | None = None) -> HookFn:
    """Mean-ablate: replace with a dataset mean rather than zero.

    Preferred over zero ablation, because zero is off-distribution and its effect
    conflates "this component mattered" with "this activation was far from zero".
    """
    return patch_hook(mean, index)
