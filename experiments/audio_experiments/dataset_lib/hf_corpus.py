"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Access to the published corpus on the Hugging Face Hub.

The Hub dataset holds **one row per conversation**: the whole session as a single
mixed render, with per-turn speaker, text and onset travelling alongside it in
``turns``. Turn *i* is therefore the audio between its own onset and the next
one, and the last turn runs to the end of the session.

Requires revision **v1.0.1 or later**. v1.0.0 published turn times that were
planned during text generation rather than measured from the audio, so slicing a
session by them yields the wrong audio for most turns; see
``dataset/scripts/recover_turn_onsets.py`` in the repository root.

Decoding a session is far more expensive than slicing it, so decoded sessions are
kept in a small LRU. Iterating turns in conversation order (as the pre-caching
pass does) then costs one decode per conversation rather than one per turn.
"""

from __future__ import annotations

from collections import OrderedDict
from math import gcd

import numpy as np
import torch

# Cached answer from _have_torchcodec; probing costs a decode, and every
# DataLoader worker would otherwise repeat it.
_TORCHCODEC_OK = None

DEFAULT_REPO_ID = "TCLResearchEurope/intelligent_wakeup"
DEFAULT_REVISION = "v1.0.1"

# Speaker name that marks the assistant's own turns.
ASSISTANT_SPEAKER = "Sigma"

# Slices shorter than this carry no utterance; two onsets landed on top of
# each other. Returned as a short run of silence rather than an empty tensor,
# which would break the encoder's feature extractor.
MIN_SLICE_SECONDS = 0.05


def load_split(repo_id: str, revision: str, split: str, sample_rate: int):
    """Load one split of the corpus, decoded at the rate the encoder wants.

    Args:
        repo_id: Hub dataset id.
        revision: Tag, branch or commit. Use v1.0.1 or later.
        split: ``train``, ``validation`` or ``test``.
        sample_rate: Rate to decode audio at.

    Returns:
        A ``datasets.Dataset``.

    Raises:
        SystemExit: If ``datasets`` is not installed.
    """
    # Imported lazily so the module can be inspected without datasets installed,
    # and so the error below is the one users actually see.
    # pylint: disable=import-outside-toplevel
    try:
        from datasets import Audio, load_dataset
    except ImportError:  # pragma: no cover - environment problem, not logic
        raise SystemExit(
            "datasets is required: pip install 'datasets>=4.0' torchcodec"
        ) from None

    dataset = load_dataset(repo_id, revision=revision, split=split)

    # datasets 4.0+ decodes audio through torchcodec, which links against
    # FFmpeg. Where that is unavailable -- an unprivileged container with no way
    # to apt-get it, say -- fall back to handing back the raw bytes and letting
    # soundfile decode them. The corpus is plain RIFF/WAVE, which soundfile
    # reads with no system libraries at all, so this costs only the resampling
    # that cast_column would otherwise have done.
    if _have_torchcodec():
        # Channel count is deliberately left alone: the keyword for it was
        # renamed between datasets 4.x and 5.x, the corpus is mono anyway, and
        # decode_session mixes down whatever it is handed.
        return dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
    return dataset.cast_column("audio", Audio(decode=False))


def _have_torchcodec() -> bool:
    """Whether torchcodec can actually decode, not merely be imported.

    Importing it succeeds even when its FFmpeg libraries are missing: the
    shared objects are loaded lazily by the first decoder. Checking the import
    therefore reports success and the failure surfaces much later, inside a
    DataLoader worker, as "Could not load libtorchcodec". So decode something
    real -- a one-frame WAV built here -- and see whether it works.

    Returns:
        True when torchcodec is usable.
    """
    global _TORCHCODEC_OK  # pylint: disable=global-statement
    if _TORCHCODEC_OK is not None:
        return _TORCHCODEC_OK

    _TORCHCODEC_OK = False
    try:
        from torchcodec.decoders import (  # pylint: disable=import-outside-toplevel
            AudioDecoder,
        )

        AudioDecoder(_silent_wav()).get_all_samples()
        _TORCHCODEC_OK = True
    except Exception:  # pylint: disable=broad-except
        pass
    return _TORCHCODEC_OK


def _silent_wav(frames: int = 256, rate: int = 16000) -> bytes:
    """Build a minimal mono 16-bit WAV in memory, for the decoder probe.

    Args:
        frames: Sample count.
        rate: Sample rate.

    Returns:
        WAV file bytes.
    """
    import struct  # pylint: disable=import-outside-toplevel

    data = b"\x00\x00" * frames
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


def decode_session(value) -> tuple[np.ndarray, int]:
    """Return one row's audio as mono float32 samples plus its rate.

    ``datasets`` 4.0 hands back a torchcodec ``AudioDecoder``; older versions
    hand back a dict. Both are accepted so the experiment runs on either.

    Args:
        value: The row's ``audio`` field.

    Returns:
        Tuple of (samples, sample rate).
    """
    if isinstance(value, dict):
        # Undecoded, from the soundfile fallback in load_split.
        if "bytes" in value:
            import io  # pylint: disable=import-outside-toplevel
            import soundfile  # pylint: disable=import-outside-toplevel

            data, rate = soundfile.read(
                io.BytesIO(value["bytes"]), dtype="float32", always_2d=True
            )
            return data.mean(axis=1), int(rate)
        # The pre-4.0 decoded form.
        return np.asarray(value["array"], dtype=np.float32), int(value["sampling_rate"])

    samples = value.get_all_samples()
    data = samples.data
    if hasattr(data, "numpy"):
        data = data.numpy()
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:  # (channels, samples)
        data = data.mean(axis=0)
    return data, int(samples.sample_rate)


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio with a polyphase filter.

    Only used when the decoder hands back a rate other than the one asked for;
    normally ``cast_column`` has already done the work.

    Args:
        samples: Float sample array.
        source_rate: Rate the samples are at.
        target_rate: Rate wanted.

    Returns:
        Resampled float32 array.
    """
    if source_rate == target_rate or len(samples) == 0:
        return samples
    # Only this fallback path needs SciPy; cast_column normally resamples.
    from scipy.signal import resample_poly  # pylint: disable=import-outside-toplevel

    divisor = gcd(int(source_rate), int(target_rate))
    return resample_poly(
        samples, target_rate // divisor, source_rate // divisor
    ).astype(np.float32)


class SessionAudio:
    """Decodes conversations on demand and cuts turns out of them.

    Args:
        dataset: The loaded split.
        sample_rate: Rate the encoder expects.
        cache_size: How many decoded sessions to keep. One is enough when turns
            are visited in conversation order; raise it for shuffled access.
    """

    def __init__(self, dataset, sample_rate: int, cache_size: int = 2):
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.cache_size = max(1, cache_size)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def session(self, row_index: int) -> np.ndarray:
        """Return the decoded samples for one conversation.

        Args:
            row_index: Row index within the split.

        Returns:
            Mono float32 samples at ``self.sample_rate``.
        """
        cached = self._cache.get(row_index)
        if cached is not None:
            self._cache.move_to_end(row_index)
            return cached

        samples, rate = decode_session(self.dataset[row_index]["audio"])
        if rate != self.sample_rate:
            samples = resample(samples, rate, self.sample_rate)

        self._cache[row_index] = samples
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return samples

    def turn(self, row_index: int, start: float, end: float) -> torch.Tensor:
        """Cut one turn out of its conversation.

        Args:
            row_index: Row index within the split.
            start: Turn onset in seconds.
            end: Where the turn ends -- the next onset, or the session's end.

        Returns:
            Mono float32 tensor of the turn's audio.
        """
        samples = self.session(row_index)
        first = max(0, int(start * self.sample_rate))
        last = min(len(samples), int(end * self.sample_rate))
        if last - first < MIN_SLICE_SECONDS * self.sample_rate:
            return torch.zeros(int(MIN_SLICE_SECONDS * self.sample_rate))
        return torch.from_numpy(np.ascontiguousarray(samples[first:last]))


def turn_spans(turns: list[dict], duration: float) -> list[tuple[float, float]]:
    """Return each turn's (start, end) in seconds.

    A turn runs until the next one starts; the last runs to the end of the
    session, trailing ambience included.

    Args:
        turns: The row's ``turns`` list, in order.
        duration: Session duration in seconds.

    Returns:
        One (start, end) pair per turn.
    """
    starts = [float(turn["time"]) for turn in turns]
    ends = starts[1:] + [float(duration)]
    return list(zip(starts, ends))
