# Voice-Diversity Analysis — Results

Full-scale run per `experiment-setup.md`, across 4 systems: 169 ElevenLabs voices
(8 utterances each, 1,352 total), the full VCTK natural reference (109 speakers — its entire
corpus, 8 utterances each, 872 total), 100 Qwen3-TTS-cloned LibriTTS speakers (8 utterances
each, 800 total, see "3rd system"), and 169 Qwen3-TTS voice-*designed* voices (same characters
and text as the ElevenLabs system, 8 utterances each, 1,352 total, see "4th system") — embedded
with two independent speaker encoders (ECAPA-TDNN, WavLM-SV).

All artifacts referenced below live under `experiments/tts_voice_analysis/output/`:
- `analysis/` — every summary JSON, plot, and table cited here
- `diversity/`, `vctk/`, `qwen/`, `qwen_designed/` — the raw audio corpora
- `embeddings/`, `vctk_embeddings/`, `qwen_embeddings/`, `qwen_designed_embeddings/`,
  `*_normalized/` — extracted speaker embeddings
- `acoustic_features/{elevenlabs,qwen_cloned,qwen_designed,natural}.csv` — per-utterance acoustic
  features (F0, HNR, jitter/shimmer, formants, bandwidth, SNR, ...)
- `pipeline_provenance.json` — git commit + config hash + achieved sample sizes for this run

## Reproduce these results

```bash
git checkout <git_commit from output/pipeline_provenance.json>
./experiments/tts_voice_analysis/pipeline/run_pipeline.sh
```

(If `git_commit` predates the `lib`/`generate`/`channel`/`features`/`analyze`/`pipeline` package
split, use that commit's own flat script layout instead — `git_commit` pins the code as it
existed then, not the current file layout.)

`pipeline/pipeline_config.env` is the single source of truth for every parameter used (utterance
counts, VCTK/LibriTTS speaker targets, encoders, perturbation subset size) — `run_pipeline.sh`
sources it and runs all 26 sub-steps (including both Qwen3-TTS systems and the voice-quality
comparison across all 4). `QWEN_BASE_URL` points
at whichever server was live when a run happened - the IP changed once already
(2026-08-12); each Qwen run's own `provenance.json` records the URL it actually used, so
historical runs stay traceable even after the config's default moves on.
`output/pipeline_provenance.json` records exactly which git commit and config hash produced a
given `output/` (plus the VCTK/LibriTTS speaker counts actually achieved — 109/150 and 100/100
respectively, since VCTK caps out at its real speaker count). **This only pins the _data and
config_ inputs** — LLM text and ElevenLabs/Qwen3-TTS audio are not bit-exact deterministic, so a
rerun reproduces the same statistical picture, not byte-identical files. Every stage's caching is
also content-hash-based (see `lib/content_cache.py`): editing `voice_mapping.json` and rerunning only
regenerates what actually changed. It also only works once these scripts are themselves committed
to git; until then, `git_commit` describes the rest of the repo but not this experiment's code.
Reaching Qwen3-TTS additionally requires the internal server (`QWEN_BASE_URL` in the config) to
be reachable - it isn't a public API.

To reproduce (or further tweak) only the natural reference, without regenerating the 169
voices:
```bash
cd experiments/tts_voice_analysis
python3 -m generate.vctk_reference \
    --n-speakers 150 --n-utterances 8 --revision 73ef4ee7d49a6fed4ea1efd65f82b4c95faeb9de
```
then re-run steps 4-5, 10, 12-13, 15-16 from `README.md` (everything downstream of the
natural corpus) — ElevenLabs-voice artifacts don't need to change.

## Core diversity claim (doc steps 1–4)

| | ECAPA-TDNN | WavLM-SV |
|---|---|---|
| Intra-speaker mean distance (ElevenLabs) | 0.372 | 0.075 |
| Inter-speaker mean distance (ElevenLabs) | 0.759 | 0.325 |
| Intra-speaker mean distance (Natural (VCTK), n = 109) | 0.246 | 0.053 |
| Inter-speaker mean distance (Natural (VCTK), n = 109) | 0.845 | 0.291 |
| Separation ratio, ElevenLabs (inter/intra) | 2.04 | 4.33 |
| Separation ratio, Natural (VCTK) | 3.43 | 5.49 |
| Nearest-centroid identification accuracy | **99.3%** | **72.1%** |
| Voices with a "too-close" nearest neighbor | **135/169 (80%)** | 81/169 (48%) |

Source: `analysis/{ecapa,wavlm_sv}_summary.json`, `analysis/{ecapa,wavlm_sv}_distributions.png`,
`analysis/{ecapa,wavlm_sv}_nearest_neighbor.json`.

**Reading these numbers:**
- Separation is real and statistically solid on both encoders (inter-speaker ≫ intra-speaker in
  every case).
- But it's weaker than natural speakers' own separation (ratio 2.04–4.33 vs. natural's 3.43–5.49).
- The two independent encoders disagree substantially on the fine-grained picture: 99.3% vs. 72.1%
  identification accuracy, 80% vs. 48% of voices flagged with an unusually close neighbor. Both
  agree separation exists on average; they disagree on how distinct each individual voice is.
- The "too-close neighbor" threshold is now calibrated off the full 109-speaker natural corpus
  (up from an initial 30-speaker pass — see git history), so its 5th-percentile value is a much
  more reliable calibration point (~5-6 speakers in the tail rather than ~1-2). The numbers barely
  moved between the 30- and 109-speaker runs (e.g. ECAPA collisions 136→135), which is itself a
  useful stability check: the original small-sample result wasn't a calibration artifact.
- The collision metric compares ElevenLabs-voice nearest-neighbor distances (centroid-to-centroid)
  against the natural corpus's own nearest-neighbor-centroid distribution (also centroid-to-centroid) —
  consistent granularity on both sides of the comparison.

## All 4 systems together

The single table every number below is drawn from. Source: `analysis/multi_system_combined.json`,
`analysis/{ecapa,wavlm_sv}_multi_system_summary.json`, `analysis/{ecapa,wavlm_sv}_multi_system_distributions.png`.

| | ECAPA-TDNN | | | | WavLM-SV | | | |
|---|---|---|---|---|---|---|---|---|
| | ElevenLabs | Qwen (Cloned) | Qwen (Designed) | Natural (VCTK) | ElevenLabs | Qwen (Cloned) | Qwen (Designed) | Natural (VCTK) |
| n voices/speakers | 169 | 100 | 169 | 109 | 169 | 100 | 169 | 109 |
| Intra-speaker mean distance | 0.372 | 0.321 | 0.644 | 0.246 | 0.075 | 0.052 | 0.139 | 0.053 |
| Inter-speaker mean distance | 0.759 | 0.698 | 0.517 | 0.845 | 0.325 | 0.285 | 0.302 | 0.291 |
| Separation ratio | 2.04 | 2.17 | **0.80** | 3.43 | 4.32 | 5.47 | 2.16 | 5.53 |
| Identification accuracy | 99.3% | **100.0%** | 59.8% | 100.0% | 72.1% | 96.2% | **17.1%** | 95.3% |
| "Too-close" neighbor rate | 135/169 (80%) | 69/100 (69%) | **169/169 (100%)** | 7/109 (6%) | 81/169 (48%) | 10/100 (10%) | 166/169 (98%) | 6/109 (6%) |

- **ElevenLabs** — 169 ElevenLabs voices, `voice_mapping.json` descriptions, full-scenario `voice_id`
- **Qwen (Cloned)** — 100 Qwen3-TTS voices cloned (ICL mode) from LibriTTS-R reference clips
- **Qwen (Designed)** — 169 Qwen3-TTS voices from `/generate-voice-design`, no reference audio
- **Natural (VCTK)** — the full 109-speaker VCTK corpus

## 3rd system: Qwen3-TTS voices cloned from LibriTTS speakers

**Note on methodology provenance:** this was originally framed as "replicate the speaker-diversity
probing from Cocktail-Talker (arXiv:2607.27756)," but that paper doesn't run a diversity
evaluation at all — Qwen3-TTS + LibriTTS appear there only as a one-line detail in its synthetic
*dialog-training-data* pipeline ("the other speakers' voices are randomly sampled from LibriTTS"),
with no embedding analysis, distance metrics, or fixed methodology to replicate. What follows is
our own methodology (identical to the ElevenLabs-vs-VCTK comparison above) applied to a 3rd
system, using the same paper's voice-sourcing idea as inspiration.

**Setup:** 100 LibriTTS-R (`train-clean-100`, CC BY 4.0) speakers, one 3-10s reference clip each,
cloned via Qwen3-TTS's ICL mode (exact reference transcript passed as `ref_text`, per the API's
own guidance that this gives better quality than x-vector-only mode) onto 8 fresh LLM-written,
persona-free sentences per speaker — same generation approach as the ElevenLabs diversity corpus,
distinct text. Qwen3-TTS calls ran strictly sequentially against a shared internal GPU server.

**Reading these numbers — this is the most informative addition to the study:**
- On WavLM-SV, Qwen (Cloned) voices (separation ratio 5.47, 96.2% ID accuracy, 10% collision rate)
  are nearly indistinguishable from the Natural (VCTK) baseline (5.53, 95.3%, 6%) — cloning real,
  distinct human speakers preserves identity very well under this encoder, as expected.
- But even genuine voice cloning of *known-to-be-distinct* real speakers still shows a real
  collision rate under ECAPA (69/100, 69%) — far above natural speech's own 6%, though still
  meaningfully below ElevenLabs generation's 80%. Since the Qwen (Cloned) system's underlying
  identities are provably different real humans, this collision rate can't be explained by the
  source speakers actually sounding alike — it points at neural TTS/cloning pipelines in general
  (not just ElevenLabs's voice generation specifically) compressing some speaker-identity information that
  ECAPA is sensitive to, at least for short utterances like these.
- Net effect: this reframes part of the "80% too-close" ElevenLabs finding. It's not fully
  explained by a general TTS-synthesis effect (Qwen is measurably better: 69% vs 80% on ECAPA,
  and much better on WavLM-SV: 10% vs 48%) — so ElevenLabs's *novel*-voice generation does appear
  harder than cloning *existing* voices — but neither is it a defect unique to ElevenLabs, since
  even cloning real distinct speakers doesn't reach natural speech's ~6% baseline on ECAPA.

## 4th system: Qwen3-TTS voice-designed voices — a clean failure mode

**Setup:** Qwen3-TTS's `/generate-voice-design` endpoint takes a natural-language style
instruction instead of a reference clip - no reference audio, no persistent voice object.
Each of the 169 ElevenLabs characters' `voice_mapping.json` `description` was distilled by an
LLM into a short (~12-25 word) instruction matching the endpoint's own example register
("Speak in a calm, warm, elderly female voice" — see `generate/qwen_design_instructions.py`
for why shortening was needed: the field isn't meant for a biography). The *same* 8
diversity-utterance texts used for the ElevenLabs system were then synthesized per character,
calling the endpoint 8 separate times with the identical instruction string each time - the
only way to get "multiple utterances of one voice" from a stateless, no-reference endpoint.
Numbers in the "All 4 systems together" table above.

**Why shorten at all? (unofficial A/B check, not part of the pipeline above)** The distillation
step was originally a hypothesis - long descriptions mix real vocal cues with irrelevant
biography that could dilute the style instruction. To check this directly, 20 characters were
also voice-designed using their full, unshortened `voice_mapping.json` description as `instruct`
verbatim (same 8 texts, same everything else), and compared against the official short-instruction
run restricted to the same 20 characters:

| variant | encoder | intra-speaker | inter-speaker | sep. ratio | ID accuracy |
|---|---|---|---|---|---|
| full description | ECAPA-TDNN | 0.636 | 0.481 | 0.76 | 80.0% |
| shortened (official) | ECAPA-TDNN | 0.600 | 0.528 | 0.88 | 84.4% |
| full description | WavLM-SV | 0.130 | 0.276 | 2.12 | 45.0% |
| shortened (official) | WavLM-SV | 0.115 | 0.331 | 2.88 | 53.1% |

The full-description variant is worse on both encoders - lower inter-speaker separation, lower
separation ratio, lower identification accuracy - confirming the shortening step is doing real
work, not just tidying: Qwen's voice-design instruction-following degrades with longer, more
narrative prompts, diluting the acoustically-relevant cues (pitch, pace, timbre, accent) that a
short, targeted instruction keeps front and center. (Script:
`unofficial/compare_design_instruction_length.py`; not part of `run_pipeline.sh`.)

**Reading these numbers — this is a clean failure mode, not a subtle nuance:**
- On ECAPA, the separation ratio is **below 1**: the 8 utterances nominally "of the same voice"
  are on average *more* spread out from each other than two entirely different voices are. That
  is the opposite of what identity separation should look like.
- On WavLM-SV, average separation is nominally positive (2.16) but per-utterance identification
  accuracy collapses to 17.1% - individual utterances land closer to a *different* character's
  centroid far more often than not. A positive average gap doesn't help if intra-speaker spread is
  wide enough to blanket most of the inter-speaker distribution.
- This isn't evidence that Qwen3-TTS voice design is broken - it's the predictable consequence of
  calling a stateless, reference-free endpoint repeatedly with the same short text instruction and
  expecting a persistent identity to fall out. Every other system in this study (ElevenLabs
  `voice_id`, Qwen ICL cloning from a reference clip, natural speech) has *something* that anchors
  "the same voice" across separate recordings; voice-design mode as used here has nothing but a
  short sentence re-interpreted independently each call.
- Practical implication: this rules voice-design mode out as a way to generate many *distinct,
  individually re-synthesizable* characters unless paired with something that pins the voice down
  (e.g. reusing a returned seed/reference if the API ever exposes one, or falling back to
  ICL-style cloning from the first design call's own output as a reference for subsequent calls).

## Channel-confound controls (doc steps 5–7) — clean pass

### Step 5: channel perturbation

Every one of 7 perturbations (bandwidth limiting, EQ tilt, GSM/Opus codec degradation, additive
noise at 15dB/5dB SNR, reverb) shifts a voice's embedding *less* than switching identity does, on
both encoders, no exceptions — e.g. ECAPA's worst case (noise at 5dB SNR, mean distance 0.340) still
sits below the inter-speaker baseline (0.759).

Source: `analysis/{ecapa,wavlm_sv}_channel_perturbation_summary.json`,
`analysis/{ecapa,wavlm_sv}_channel_perturbation.png`.

### Step 6: channel-normalized replication

Separation ratio retained after conservative loudness + bandwidth normalization:

| | ECAPA-TDNN | WavLM-SV |
|---|---|---|
| ElevenLabs separation retained | 99.8% | 100.0% |

Unnormalized/normalized numbers are near-identical to 3 decimal places — normalization essentially
doesn't touch the result, the strongest possible outcome for this control.

Source: `analysis/{ecapa,wavlm_sv}_channel_normalization_summary.json`,
`analysis/{ecapa,wavlm_sv}_channel_normalization.png`.

### Step 7: acoustic characterization + confound check

Feature means, ElevenLabs vs. Natural (VCTK) (n=1,352 vs. n=872 utterances):

| feature | ElevenLabs | Natural (VCTK) |
|---|---|---|
| Median F0 (Hz) | 191.9 | 163.3 |
| F0 range (Hz) | 158.1 | 142.4 |
| HNR (dB) | 11.9 | 12.2 |
| Speaking rate (words/s) | 3.42 | 2.61 |
| Bandwidth (Hz) | 3417.8 | 4394.7 |
| Spectral slope (dB) | 21.1 | 18.9 |
| Spectral flatness | 0.029 | 0.018 |
| SNR (dB) | 32.6 | 14.7 |
| Clipping rate | 0.000 | 0.000 |

Confound check — Pearson correlation between pairwise embedding distance (ECAPA) and pairwise
channel-feature difference, across all 14,196 ElevenLabs-voice pairs (unchanged by the natural-corpus
extension, since this check only uses ElevenLabs-voice pairs):

| channel feature | r | p |
|---|---|---|
| Bandwidth | +0.259 | 7.5e-217 |
| Spectral flatness | +0.181 | 4.2e-105 |
| SNR | +0.175 | 1.6e-98 |
| Spectral slope | +0.095 | 6.2e-30 |
| Clipping rate | −0.038 | 5.2e-6 |

All `|r| ≤ 0.26` (below the 0.3 "weak" threshold). At n=14,196, p-values are near-zero for any
nonzero effect — **r is the number that matters here, not p.**

Source: `analysis/acoustic_confound_summary.json`, `analysis/acoustic_confound_correlations.png`.

### Step 7b: voice-quality/prosody across all 4 systems

Jitter, shimmer, formants (F1-F3), pitch coefficient of variation (f0_cv) and voicing fraction,
compared across ElevenLabs (n = 1,352), Qwen (Cloned) (n = 800), Qwen (Designed) (n = 1,352) and
Natural (VCTK) (n = 872). Unlike the step 7 features above, these aren't channel/recording-quality
cues - they're voice-quality and prosody descriptors that can help explain *why* the diversity
numbers elsewhere in this doc look the way they do.

| feature | ElevenLabs | Qwen (Cloned) | Qwen (Designed) | Natural (VCTK) |
|---|---|---|---|---|
| Jitter (%) | 2.20 | 2.18 | 2.10 | 2.54 |
| Shimmer (%) | 10.14 | 7.48 | 6.95 | 9.39 |
| F1 (Hz) | 522 | 400 | 481 | 463 |
| F2 (Hz) | 1623 | 1641 | 1684 | 1662 |
| F3 (Hz) | 2673 | 2748 | 2734 | 2730 |
| F0 coefficient of variation | 0.343 | 0.324 | 0.271 | 0.410 |
| Voicing fraction | 0.566 | 0.567 | 0.602 | 0.422 |

- **Jitter/shimmer don't tell one clean story.** All three synthetic systems have *lower* jitter
  than natural speech (2.10-2.20% vs. 2.54%) - consistent with the classic "TTS is too
  pitch-stable" tell. But shimmer splits: ElevenLabs (10.14%) actually has *more* amplitude
  perturbation than Natural (VCTK) (9.39%), while both Qwen modes are noticeably smoother
  (6.95-7.48%). So "unnaturally clean" voice quality is a Qwen-cloning/design characteristic
  here, not a universal synthetic-speech artifact - it doesn't show up in ElevenLabs at all.
- **Formants: the real gap is in F1, and it's specific to Qwen cloning.** F2 and F3 are close
  across all 4 systems (within ~120 Hz of each other). F1 is where a system stands out:
  Qwen (Cloned) sits at 400 Hz, well below Natural (VCTK) (463 Hz) and both ElevenLabs (522 Hz) and
  Qwen (Designed) (481 Hz) - a systematic downward shift in first-formant frequency, i.e. the
  cloning pipeline is measurably compressing vocal-tract/vowel-openness cues relative to the
  LibriTTS speakers it's cloning from. ElevenLabs's F1 spread is also the widest (std 105 Hz vs.
  57-100 Hz elsewhere), consistent with it being the system explicitly designed for maximum
  voice diversity.
- **Pitch coefficient of variation gives a clean, monotonic ranking:** Natural (VCTK) (0.410) >
  ElevenLabs (0.343) > Qwen (Cloned) (0.324) > Qwen (Designed) (0.271). Voice-design mode is the
  flattest/most monotone system of the four - consistent with the 4th-system "clean failure
  mode" finding above (no reference audio, no persistent identity): it's not just inconsistent
  *across* utterances of the same nominal voice, it's also less intonationally expressive
  *within* any single utterance.
- **Voicing fraction is the single largest, most consistent gap in this whole feature set.**
  All three synthetic systems pack in substantially more continuous voicing (0.566-0.602) than
  natural speech (0.422) - real VCTK recordings spend roughly 30-40% more time in silence/unvoiced
  segments per utterance than any TTS system tested. This holds regardless of whether the voice
  comes from a persistent voice_id (ElevenLabs), reference-audio cloning, or a style instruction
  (Qwen design) - suggesting it's a property of how these TTS systems pace and pause speech in
  general, not of any one voice-identity mechanism.

Source: `analysis/voice_quality_multi_system.json`,
`analysis/voice_quality_{jitter_shimmer,formants,pitch_cv,voiced_fraction}.png/.pdf`.

## Overall verdict

The harder, more rigorous claim — that voice separation isn't a channel/recording-quality
artifact — **passed cleanly on every control**: perturbations move embeddings less than identity
changes, separation survives normalization untouched, and channel-feature correlations with
embedding distance are all weak.

The core "150+ distinct identities" claim is real but more nuanced than a flat success: voices are
well-separated on average and this isn't explained by recording quality, but a meaningful fraction
(especially by ECAPA) sit closer to a neighbor than is typical among distinct natural speakers, and
the two encoders disagree noticeably on severity. This held up after recalibrating against the full
109-speaker natural corpus rather than an initial 30-speaker sample. Worth reflecting that nuance in
the paper rather than claiming all voices are fully and uniformly distinct.

The Qwen3-TTS/LibriTTS 3rd-system comparison adds useful context here: cloning genuinely distinct
real speakers *also* produces a non-trivial collision rate (69% on ECAPA, vs. natural speech's 6%)
— well below ElevenLabs's 80%, but still far from natural. This suggests part of the phenomenon is
a general property of neural voice synthesis/cloning under short-utterance embedding comparison,
not a defect unique to this system's voice-generation approach, even though ElevenLabs's *novel*-voice
generation does still measurably underperform Qwen's *cloning* of existing voices.

The 4th system (Qwen voice-design, no reference audio) is the sharpest data point in the whole
study: separation ratio below 1 on ECAPA, 17.1% identification accuracy on WavLM-SV - a genuine
collapse, not a nuance. It's also the most useful negative control available: it confirms that
this analysis pipeline *can* detect a real absence of speaker identity when one is actually
absent, rather than every system just scoring "good" by construction. That ElevenLabs, Qwen
cloning, and natural speech all land in a completely different regime (95-100% identification
accuracy on at least one encoder each) is stronger evidence for the core diversity claim than any
single system's numbers alone - the method has demonstrated it can fail loudly when the underlying
premise (a persistent per-character voice) doesn't hold.
