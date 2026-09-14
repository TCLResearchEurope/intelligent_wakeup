# Offline Wakeup Detector

A trained baseline for device-directed speech detection: given a turn and the
conversation before it, decide whether the assistant should respond.

Where `experiments/gpt_realtime` probes a general-purpose realtime model with a
prompt, this trains a small model for the task — a frozen Whisper encoder per
turn, then a causal transformer over the turn sequence.

Data comes from the published corpus,
[`TCLResearchEurope/intelligent_wakeup`](https://huggingface.co/datasets/TCLResearchEurope/intelligent_wakeup).
Nothing local is needed; `datasets` fetches and caches it.

## Setup

```bash
micromamba create -n intelligent_wakeup_audio_experiments -f environment.yml
micromamba activate intelligent_wakeup_audio_experiments
```

`torchcodec` decodes the corpus audio and links against FFmpeg. Install it from
the same index as torch — the PyPI wheel links against a CUDA build and will not
load beside a CPU one:

```bash
pip install torch torchcodec --index-url https://download.pytorch.org/whl/cpu
```

On conda take FFmpeg from conda-forge (as `environment.yml` does) rather than
the system: conda ships its own `libstdc++`, and a system FFmpeg built against a
newer one fails to load.

## Running

Three steps. The first is by far the slowest and only has to happen once.

```bash
# 1. encode every turn once through the frozen Whisper encoder
python precompute_embeddings.py                   # train + validation
python precompute_embeddings.py 'splits=[test]'   # add test when you need it

# 2. train (Phase 1: encoder frozen, current turn read from the cache)
python train.py

# 3. threshold sweep on a checkpoint
python evaluate.py checkpoint=checkpoints/epoch018_f10.4710.pt
python evaluate.py checkpoint=... eval_on=test
python evaluate.py checkpoints=[a.pt,b.pt]   # average several into one score
```

Phase 2 unfreezes the top Whisper encoder layers and reads raw audio for the
current turn, at a lower learning rate:

```bash
python train.py model.unfreeze_top_layers=4 training.phase2_start_epoch=40
```

Any config value overrides on the command line:

```bash
python train.py training.batch_size=64 training.num_epochs=80
python train.py device=cuda
python precompute_embeddings.py dataset.embedding_cache_dir=.cache/embeddings_v2
```

In zsh, quote anything with brackets: `'splits=[test]'`.

### What to expect

| step | scale | time (12 CPU threads) |
|---|---|---|
| precompute, validation | 202 conversations, 3,103 turns | ~23 min |
| precompute, train | 1,852 conversations, 28,717 turns | ~3 h |
| corpus download (once) | all splits | ~20 GB |
| embedding cache | 512 floats per turn | ~2 GB |

A GPU turns the precompute from hours into minutes; it is pure forward passes
through `whisper-base`. See "Training on the cluster" below.

Two things that look like failures but are not:

- A `PyGILState_Release` crash dump **after** the script reports success. It is
  torch/torchcodec noise at interpreter teardown; the exit code is still 0.
  Check the cache count rather than the tail of the log.
- `train.py` refusing to start. It checks embedding-cache coverage first and
  tells you which precompute command to run, rather than training on zeros.

## Training on the cluster

Runs on the internal k8s cluster through `nihao`:

```bash
nihao make wakeup-train -f experiments/audio_experiments
```

nihao copies this directory to
`/nas/projects/voice-assistant-for-tvs/models/<user>/wakeup/train`
and runs `nihao.sh` there on one GPU: precompute, then training. Watch it with
`nihao jobs` / `nihao pods`, or `nihao-ui`.

### The image

`docker/Dockerfile` builds the environment — CUDA torch, torchcodec, FFmpeg and
the Python dependencies. It carries **no code**: nihao supplies that by copying
the directory, so a code change needs no rebuild and only a dependency change
does.

```bash
# once
docker buildx create --name iw.buildx --use

docker buildx build --builder=iw.buildx \
  -t gitlab-registry.tcl-research.pl/tcl-research/voice-assistant-for-tvs/intelligent_wakeup/intelligent-wakeup-train:base-v0.1.0 \
  --provenance false \
  --push \
  -f experiments/audio_experiments/docker/Dockerfile \
  experiments/audio_experiments
```

Three things this command encodes, each of which fails without it:

- **`gitlab-registry.tcl-research.pl`, not `registry.tcl-research.pl`.** The
  latter is read-only and answers a push with `Method not allowed`.
- **`--provenance false`.** GitLab's registry rejects the attestation manifest
  buildx attaches by default.
- **No `sudo`.** Credentials live in `~/.docker/config.json`; under sudo, Docker
  reads root's instead and the push is unauthenticated. Either
  `sudo docker login <host>` or add yourself to the `docker` group.

The registry path mirrors the GitLab project path, and **the project must be
unarchived to push** — archived projects are read-only, registry included. Pulls
still work once it is archived again, so it is rebuilding that needs the project
active, not running a job.

Then point `image:` in `nihao-config.yml` at the tag. Bump the tag when
`requirements.txt` changes, the way `md-qwen3-tts` versions its base image.

The build order matters: torch and torchcodec are installed together from the
CUDA index *before* `requirements.txt`, because the torchcodec wheel on PyPI
links against a different torch build and fails at import. Installing them first
leaves the `torchcodec` pin already satisfied, so pip will not pull the PyPI
wheel over the top.

Without a prebuilt image you can point `image:` at any CUDA base and have
`nihao.sh` install at start-up, but that costs minutes on every run.

### Where things are written

nihao **deletes and re-copies the working directory on every deploy**, and does
not copy hidden files. So nothing slow to produce may live there — it all goes
under `IW_WORK_DIR` on `/nas`:

```
/nas/projects/<project>/models/<user>/intelligent-wakeup/
  hf/                       HF_HOME — the ~20 GB corpus, downloaded once
  embeddings/               embedding cache, ~2 GB
  checkpoints/<timestamp>/  one directory per run
```

Both steps are idempotent against that cache, so a second deploy skips the
download and the encoder pass and goes straight to training. This is also why
`training.checkpoint_dir` is overridden: the config default is relative to the
working directory and the next deploy would delete it.

### Knobs

| variable | default | |
|---|---|---|
| `IW_WORK_DIR` | set in `nihao-config.yml` | persistent root: corpus, embeddings, checkpoints |
| `IW_SPLITS` | `[train,validation]` | which splits to encode |
| `IW_EPOCHS` / `IW_LR` / `IW_BATCH` | config | the usual training knobs |
| `IW_POS_WEIGHT` | config | positive class weight |
| `IW_UNFREEZE` / `IW_PHASE2_START` / `IW_PHASE2_LR` | config | Phase 2 fine-tuning |
| `IW_SESSION_CACHE` | config | decoded sessions held in memory |
| `IW_WAKE_SOURCE` | config | `oracle` \| `none` — see Labels |
| `IW_REQUIRE_GPU` | `1` | fail fast rather than fall back to CPU |

```bash
nihao make wakeup-train -f experiments/audio_experiments -e IW_EPOCHS=80 -e IW_LR=1e-4
nihao make wakeup-train -f experiments/audio_experiments -n vram=24GB
```

> **Why one variable per override, rather than a single `IW_TRAIN_ARGS`?**
> nihao's `-e` cannot carry a value containing `=`. `make.py:53` splits the
> entry on *every* `=` and rejoins the remainder without them, so
> `-e IW_TRAIN_ARGS='a=1 b=2'` reaches the pod as `a1 b2` and Hydra rejects it
> with `missing EQUAL at '<EOF>'`. One `=` per `-e` survives intact, hence the
> discrete knobs above. `IW_TRAIN_ARGS` still works if you set it in the `env:`
> block of `nihao-config.yml`, which does not go through that parser.

### Phase 2 fine-tuning

```bash
nihao make wakeup-ft -f experiments/audio_experiments \
  -e IW_UNFREEZE=4 -e IW_PHASE2_START=20 -e IW_PHASE2_LR=1e-5 -e IW_SESSION_CACHE=24
```

Phase 2 reads **raw audio** rather than cached embeddings, so it needs a GPU —
on CPU the encoder forward makes an epoch take hours instead of minutes. Raise
`IW_SESSION_CACHE`: training shuffles, so a batch of 64 touches ~64 different
sessions, and the default LRU of 2 would thrash and spend all its time decoding.

`project:` is `voice-assistant-for-tvs`, **not** `intelligent_wakeup`, even
though the latter exists on the NAS. The project name also becomes part of the
Kubernetes object name, and RFC 1123 forbids underscores there; nihao only
translates dots, so an underscored project is rejected with a 422 before the
job is ever created. The hyphenated group the repo lives under works and
already exists.

Only the code copy lands there. `IW_WORK_DIR` — corpus, embedding cache,
checkpoints — is pinned separately to `/nas/projects/intelligent_wakeup/`,
where the dataset archives already live.

`nihao.sh` activates the micromamba environment itself — nihao overrides the
container command, which bypasses the image's entrypoint — and **fails fast if
no GPU is visible** rather than silently falling back to CPU, where the encoder
pass alone takes about three hours.

## How turns are built

The corpus holds **one row per conversation**: the whole session as a single
mixed render, with per-turn speaker, text and onset alongside it. Turn *i* is
the audio between its own onset and the next one; the last turn runs to the end
of the session.

So a turn's audio carries the room tone, ambience and speaker overlap it was
mixed with. Earlier revisions of this experiment read clean per-utterance wavs
from a local NAS copy — those were never published, and results from them are
not comparable to results from here.

Requires dataset revision **v1.0.1 or later**. v1.0.0 published turn times that
were planned during text generation rather than measured from the audio; they
drift past the end of the session for most turns, so slicing by them yields the
wrong audio. See `dataset/scripts/recover_turn_onsets.py`.

## Labels

A turn is **expected** to trigger when the next turn is spoken by the assistant:

- **direct** — the turn contains the word "Sigma"
- **contextual** — no wake word, but a reply is expected (an open exchange)
- **non-assistance** — no reply expected; the model should stay quiet

Assistant turns are never classified, but they do serve as context.

Label balance on v1.0.1:

| split | turns | positive | negative | neg:pos |
|---|---|---|---|---|
| train | 26,136 | 2,441 | 23,695 | 9.7:1 |
| validation | 2,819 | 273 | 2,546 | 9.3:1 |
| test | 3,647 | 299 | 3,348 | 11.2:1 |

`model.pos_weight` is set to 9.7 to match. It is the single most load-bearing
hyperparameter here: too low and the loss barely penalises a missed trigger, so
the model learns to answer nothing.

## Splits

Taken from the dataset, which assigns them **by group** — a scenario's `_long`
sibling, its `_variantN` re-takes and its `no_va_` twin all stay together.

Do not re-split on conversation id. Those near-duplicates share cast, voices and
room tone, so splitting them apart puts effectively-seen audio in validation and
every number off it is optimistic. An earlier version of this experiment did a
random 15% conversation split and had exactly that problem.

## Layout

```
precompute_embeddings.py   encode every turn once, cache to disk
train.py                   Phase 1 / Phase 2 training loop
evaluate.py                threshold sweep (FA/hour) on one checkpoint or an ensemble

dataset_lib/
  hf_corpus.py             Hub loading, session decoding, turn slicing
  dataset.py               WakeupDataset — one sample per non-assistant turn
  labels.py                trigger types and speaker roles
  cache.py                 utterance id -> embedding, one file each
  collate.py               batching for both phases
  metrics.py               TP/FP/TN/FN by trigger type, shared by 2 and 3

wakeup_model/
  encoder.py               Whisper encoder + attention pooling
  context_transformer.py   causal transformer over the turn sequence
  model.py                 the detector, losses, both forward paths
  config.py                hyperparameters, mirrors configs/model/default.yaml

nihao-config.yml           cluster job definition
nihao.sh                   what the cluster job runs
docker/Dockerfile          the training environment — dependencies, no code
```

### Node selection

`chimera`'s NVIDIA driver is too old for the `torch 2.14+cu130` wheels the
virtualenv installs: CUDA reports itself unavailable and `nihao.sh` stops rather
than falling back to a CPU encoder pass that takes hours. Pin a node the
scheduler would otherwise pick at random:

```bash
nihao make wakeup-train -f experiments/audio_experiments -n pegasus
```

`pegasus` (RTX 4090, 24 GB), `dragon`/`giant` (RTX 3090, 24 GB) and
`centaur`/`siren` (RTX 5090, 32 GB) all work; `minotaur`/`medusa` (RTX 2080 Ti,
11 GB) work but only fit one Phase 2 job at a time.

Decoding a session costs far more than slicing a turn out of it, so decoded
sessions are held in a small LRU (`dataset.session_cache_size`). The precompute
walks turns in conversation order, which makes that one decode per conversation
rather than one per turn — do not shuffle it.

The embedding cache is keyed by utterance id and says nothing about where the
audio came from. **Delete `.cache/embeddings` whenever the audio source
changes** (a new dataset revision, a different sample rate), or stale entries
load silently.

### A cache and its pooling head are one artefact

What gets cached is not the Whisper output but the output of `AttentionPooling`
on top of it — and that layer is randomly initialised. Two precompute runs
therefore produce two different embedding spaces (cosine ≈ 0.99 between them,
which is enough to cost about 0.12 F1), and a model reading the wrong one is
never told.

So `precompute_embeddings.py` seeds from `cfg.seed` and writes the head it used
to `pooling_head.pt` in the cache directory. Topping the cache up reuses that
head; `train.py` loads it before training, which matters at the Phase 2 switch —
from then on the current turn is encoded by the model while the context turns
still come from the cache, and the two have to be the same space.

A cache with embeddings but no `pooling_head.pt` was built before this and its
head cannot be recovered. Phase 1 against it is fine (nothing encodes audio
there), but re-encode into a fresh directory before fine-tuning. Both scripts
say so rather than guessing.
