"""
Zipformer encoder, vendored from k2-fsa/icefall (Apache-2.0).

``zipformer.py``, ``scaling.py``, ``subsampling.py`` and
``encoder_interface.py`` are copied verbatim from
``egs/librispeech/ASR/zipformer/`` so they can be refreshed by re-copying; the
only local code is ``_shims.py``, which stands in for the ``k2`` and
``icefall`` imports. See that module for why.

Those files import each other by bare module name (``from scaling import ...``),
which only resolves when their directory is on ``sys.path`` — so this package
puts it there rather than rewriting the vendored source.
"""

import sys
from pathlib import Path

from . import _shims

_shims.install()
sys.path.insert(0, str(Path(__file__).parent))

from zipformer import Zipformer2  # noqa: E402  pylint: disable=wrong-import-position
from subsampling import (
    Conv2dSubsampling,
)  # noqa: E402  pylint: disable=wrong-import-position

from .kaldi import fbank  # noqa: E402  pylint: disable=wrong-import-position

__all__ = ["Zipformer2", "Conv2dSubsampling", "fbank"]
