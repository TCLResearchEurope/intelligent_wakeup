"""
The two things icefall's Zipformer imports that we do not want to depend on.

``zipformer.py``, ``scaling.py`` and ``subsampling.py`` are vendored verbatim
from k2-fsa/icefall (Apache-2.0) so they can be updated by re-copying. They
import two things this repo has no use for:

* **k2** — used only for the Swoosh activations, and only as a speed
  optimisation. The pure-PyTorch formulas are in those same files, in the
  ``torch.jit.is_scripting()`` branch right above each k2 call, so they are
  reproduced here rather than taking on a dependency that needs building from
  source against a matching CUDA.
* **icefall.utils.torch_autocast** — a thin wrapper over ``torch.amp.autocast``.

``kaldi.py`` is vendored the same way, from torchaudio (BSD-2), for the
Kaldi-compatible filterbank features the pretrained weights expect. torchaudio
itself is not a dependency: its wheel carries a C++ extension pinned to one
torch build, which fails to load against both the torch here and the newer one
on the cluster, while the ``compliance.kaldi`` module it contains is pure
PyTorch. Its single use of ``torchaudio.functional.create_dct`` is on the MFCC
path, not the filterbank one, and is stubbed below so the file stays a verbatim
copy.

Installed into ``sys.modules`` by this package's ``__init__``, before the
vendored files are imported.
"""

from __future__ import annotations

import math
import sys
import types
from contextlib import contextmanager

import torch
from torch import Tensor


def swoosh_l(x: Tensor) -> Tensor:
    """Swoosh-L activation.

    Args:
        x: Input tensor.

    Returns:
        ``log(1 + exp(x - 4)) - 0.08 x - 0.035``.
    """
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    return torch.logaddexp(zero, x - 4.0) - 0.08 * x - 0.035


def swoosh_r(x: Tensor) -> Tensor:
    """Swoosh-R activation.

    Args:
        x: Input tensor.

    Returns:
        ``log(1 + exp(x - 1)) - 0.08 x - 0.313261687``.
    """
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    return torch.logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687


def swoosh_l_forward_and_deriv(x: Tensor) -> tuple[Tensor, Tensor]:
    """Swoosh-L and its derivative.

    Args:
        x: Input tensor.

    Returns:
        ``(value, d value / d x)``.
    """
    return swoosh_l(x), torch.sigmoid(x - 4.0) - 0.08


def swoosh_r_forward_and_deriv(x: Tensor) -> tuple[Tensor, Tensor]:
    """Swoosh-R and its derivative.

    Args:
        x: Input tensor.

    Returns:
        ``(value, d value / d x)``.
    """
    return swoosh_r(x), torch.sigmoid(x - 1.0) - 0.08


def create_dct(n_mfcc: int, n_mels: int, norm: str | None) -> Tensor:
    """Orthonormal DCT-II matrix — stands in for ``torchaudio.functional``.

    Only the MFCC path in the vendored ``kaldi.py`` calls this; filterbank
    features do not. Implemented rather than stubbed so the vendored file stays
    usable as written.

    Args:
        n_mfcc: Number of output coefficients.
        n_mels: Number of mel bins.
        norm: ``"ortho"`` for the orthonormal scaling, else None.

    Returns:
        (n_mels, n_mfcc) transform matrix.
    """
    n = torch.arange(float(n_mels))
    k = torch.arange(float(n_mfcc)).unsqueeze(1)
    dct = torch.cos(torch.pi / float(n_mels) * (n + 0.5) * k)
    if norm is None:
        dct *= 2.0
    else:
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(n_mels))
    return dct.t()


@contextmanager
def torch_autocast(*args, **kwargs):
    """Stand-in for ``icefall.utils.torch_autocast``.

    Args:
        *args: Passed through to ``torch.amp.autocast``.
        **kwargs: Passed through; ``device_type`` defaults to cuda.

    Yields:
        None, inside the autocast context.
    """
    kwargs.setdefault("device_type", "cuda")
    with torch.amp.autocast(*args, **kwargs):
        yield


def install() -> None:
    """Put the fake ``k2`` and ``icefall`` modules into ``sys.modules``.

    Returns:
        None.
    """
    if "k2" not in sys.modules:
        k2 = types.ModuleType("k2")
        k2.swoosh_l = swoosh_l
        k2.swoosh_r = swoosh_r
        # The no-grad variants differ only in that k2 skips saving state.
        k2.swoosh_l_forward = swoosh_l
        k2.swoosh_r_forward = swoosh_r
        k2.swoosh_l_forward_and_deriv = swoosh_l_forward_and_deriv
        k2.swoosh_r_forward_and_deriv = swoosh_r_forward_and_deriv
        sys.modules["k2"] = k2

    if "torchaudio" not in sys.modules:
        torchaudio = types.ModuleType("torchaudio")
        functional = types.ModuleType("torchaudio.functional")
        functional.create_dct = create_dct
        torchaudio.functional = functional
        sys.modules["torchaudio"] = torchaudio
        sys.modules["torchaudio.functional"] = functional

    if "icefall" not in sys.modules:
        icefall = types.ModuleType("icefall")
        utils = types.ModuleType("icefall.utils")
        utils.torch_autocast = torch_autocast
        icefall.utils = utils
        sys.modules["icefall"] = icefall
        sys.modules["icefall.utils"] = utils
