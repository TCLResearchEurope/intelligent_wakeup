# int_wakeup-whisper_base-ctx20-ft4 — Results

A device-directed speech detector: a Whisper-base encoder per turn, attention
pooling, then a causal transformer over the turn sequence. 24.0M parameters.
Trained in two phases — Phase 1 with the encoder frozen and embeddings cached
(3.4M trainable), then Phase 2 with **the top 4 encoder layers unfrozen**
(16.0M trainable). The `-ft4` suffix is that second phase; it is the difference
from `int_wakeup-whisper_base-ctx20`.

Evaluated on the **test** split of
[`TCLResearchEurope/intelligent_wakeup`](https://huggingface.co/datasets/TCLResearchEurope/intelligent_wakeup)
at revision **v1.0.1**: 243 conversations, 3,647 evaluable turns, 6.72 h of
recording of which 5.28 h is speech not addressed to the assistant.

Run `checkpoints/20260914-013949`. Reported model: the three kept checkpoints
(epochs 47, 48, 49) averaged. Single-model numbers are below for comparison.

**Selection:** model and threshold were both chosen on validation and then
applied unchanged to test. Validation F1 ties at thresholds 0.30 and 0.90, so
the tie went to the lower false-accept rate: **0.90**. Had it gone to 0.30
instead, test would read F1 0.891 at 4.92 FA/hour (speech) — so the tie-break
moved false accepts, not F1. No selection of any kind was made on test.

**On the size of the validation→test gap.** Validation F1 0.896 → test 0.890 is
a small drop, and the reason is the corpus, not the model. Splits are assigned
by group, so a scenario's `_long` sibling, `_variantN` re-takes and `no_va_`
twin never straddle them and there is no leakage — but all three splits come
from one generation pipeline, one voice pool and one scenario taxonomy. Test
here means *unseen conversations, same distribution*. These numbers say nothing
about how the model behaves on real recordings.

## Headline

| | precision | recall | F1 | FA/hour (recording) | FA/hour (speech) |
|---|---|---|---|---|---|
| **ensemble of 3 @ 0.90** | **0.931** | 0.853 | **0.890** | **2.83** | **3.60** |
| best single (`epoch049`) @ 0.70 | 0.915 | 0.863 | 0.888 | 3.57 | 4.55 |
| previous model @ 0.40 | 0.824 | 0.816 | 0.820 | 7.74 | 9.90 |

**Two false-accept denominators**, because two are in circulation and they
differ by 27%. *Recording* divides by all 6.72 h, which is what an always-on
detector is actually exposed to. *Speech* divides by the 5.28 h of
not-addressed speech, which is the convention `../gpt_realtime/RESULTS.md`
uses. Compare like with like; this file gives both everywhere.

Per category at threshold 0.90:

| category | n | TP | FP | TN | FN | recall |
|---|---|---|---|---|---|---|
| overall | 3647 | 255 | 19 | 3329 | 44 | 0.853 |
| direct | 158 | 148 | — | — | 10 | 0.937 |
| contextual | 141 | 107 | — | — | 34 | 0.759 |
| non-assistance | 3348 | — | 19 | 3329 | — | — |

`direct` and `contextual` contain only positives, so their precision is
trivially 1.000 and only recall is informative. `non-assistance` contains only
negatives, so only its FP/TN counts mean anything — its F1 is structurally
0.000 and is not a score.

## What changed against the previous model

| | F1 | FA/hour (speech) | direct recall | contextual recall |
|---|---|---|---|---|
| `...-ctx20` (Phase 1 only) | 0.820 | 9.90 | 0.867 | 0.759 |
| `...-ctx20-ft4` | **0.890** | **3.60** | **0.937** | 0.759 |

**Unfreezing the encoder is what moved the numbers.** The previous model's worst
failure was that it barely used the wake word — it scored unambiguous
"Sigma, what plant is this?" at p≈0.000 while scoring human-to-human chatter at
p≈1.000, and an oracle wake-word flag did not fix it. Four unfrozen encoder
layers did: `direct` recall 0.867 → 0.937 on test, and 0.852 → 0.985 on
validation.

**`contextual` recall did not move at all** (0.759 both times). Follow-up
requests with no keyword — "and the one after that?" — are now the entire
remaining recall problem, and nothing tried so far has touched them.

**False accepts fell 64%** (9.90 → 3.60 per hour of not-addressed speech), from
two independent sources: the sharper model is more confident on the turns it
gets right, which lets the threshold move to 0.90 without losing recall; and
averaging three checkpoints cancels the confident-but-wrong predictions that a
threshold cannot separate.

## Against the GPT Realtime baseline

Same 3,647 turns, same audio. See `../gpt_realtime/RESULTS.md`.

| | F1 | recall | FA/hour (speech) |
|---|---|---|---|
| GPT Realtime (`gpt-realtime-mini`) | 0.179 | 0.997 | 516.6 |
| `...-ctx20` | 0.820 | 0.816 | 9.9 |
| **`...-ctx20-ft4`** | **0.890** | 0.853 | **3.6** |

The baseline triggers on almost everything: near-perfect recall, 2,728 false
accepts. This model gives up 14 points of recall for a 143x reduction in false
accepts.

## The model needs two encoders at inference

Phase 2 fine-tuned the encoder for the **current** turn while the **context**
turns kept coming from the cache, which was built with the **frozen** encoder.
The model expects that asymmetry. Encoding everything with the fine-tuned
encoder — the obvious way to run it outside this repo — raises nothing and
costs:

| validation, ensemble @ 0.90 | F1 | contextual recall | FA/hour (speech) |
|---|---|---|---|
| context from the frozen encoder (as trained, as reported) | 0.896 | 0.759 | 5.5 |
| context from the fine-tuned encoder | 0.718 | 0.290 | 0.95 |

Measured, not inferred: the top-4 encoder layers moved 5.3% in Phase 2, and a
context turn re-encoded by the fine-tuned model has cosine 0.59-0.82 against its
cached vector. Reconstructing the frozen encoder — pristine `openai/whisper-base`
plus the pooling head recorded in the cache — reproduces the cache at cosine
1.00000.

So a published checkpoint has to ship the context pooling head alongside the
weights, and its inference code has to run two encoders. The Hugging Face export
does both; see `hf_export/int_wakeup-whisper_base-ctx20-ft4/inference.py`.

**This is a training artefact, not a design choice**, and it is the first thing
to fix in a retrain — either re-encode the context cache with the current
encoder every few Phase 2 epochs, or accept the cost and encode context from
audio during Phase 2.

## Still open

- **<1 FA/hour is not reached.** 3.60/hour is 19 false positives on this split;
  <1 means ≤5. Nothing tried moved this below ~3.
- **`contextual` recall 0.759** — 34 missed follow-ups, unchanged across every
  experiment run so far.
- **The run was still improving when it stopped.** Phase 2 validation F1 went
  0.881 → 0.889 → 0.891 → 0.899 over its last four epochs and hit the 50-epoch
  limit, not a plateau. More Phase 2 epochs is the cheapest untried thing.

## Reproducing

The configuration is saved two ways, both named after the model:

- `configs/experiment/int_wakeup-whisper_base-ctx20-ft4.yaml` — the re-runnable
  recipe. Deltas from the defaults only, so a later change to a default is
  inherited rather than silently pinned.
- `int_wakeup-whisper_base-ctx20-ft4-config.yaml` — the byte-exact config this
  run resolved to, kept as provenance rather than as an input.

Locally, once a cache exists that this recipe built itself:

```bash
python precompute_embeddings.py 'splits=[train,validation,test]' \
  dataset.embedding_cache_dir=.cache/embeddings-v2
python train.py +experiment=int_wakeup-whisper_base-ctx20-ft4 dataset.embedding_cache_dir=.cache/embeddings-v2
```

On the cluster:

```bash
nihao make wakeup-train -f experiments/audio_experiments -n pegasus \
  -e IW_CACHE_DIR=/nas/people/<user>/intelligent_wakeup/embeddings-v2 \
  -e 'IW_SPLITS=[train,validation,test]' \
  -e IW_POS_WEIGHT=2.0 -e IW_EPOCHS=50 -e IW_PHASE2_START=35 \
  -e IW_UNFREEZE=4 -e IW_PHASE2_BATCH=8 -e IW_SESSION_CACHE=24
```

Then score it, scoring test at the threshold validation picked:

```bash
nihao make wakeup-eval -f experiments/audio_experiments -n pegasus \
  -e IW_EVAL_DIR=<checkpoint dir> -e IW_CACHE_DIR=<the same cache> \
  -e IW_UNFREEZE=4 -e IW_REPORT_THRESHOLD=0.90
```

The cache must be the one these checkpoints were trained against — it carries
the pooling head that produced it. See the pooling-head section of `README.md`.
