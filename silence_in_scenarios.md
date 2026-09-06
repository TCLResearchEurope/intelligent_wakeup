# Silence and ambient segments in scenarios

A scenario can include stretches with no dialogue, where only ambient sound plays.
These are useful for wake-word evaluation: the assistant must stay quiet through
noise that is not addressed to it.

## Declaring a silence phase

Mark a phase in the scenario config with `"silence": true`. The phase `topic`
becomes the sound description passed to the sound-effects backend:

```json
{
  "phase": "ambient_gap",
  "topic": "quiet living room, rain against the windows",
  "speakers": [],
  "silence": true,
  "duration": 6.0
}
```

A phase named `"silence"` is also treated as ambient even without the flag.

## Controlling the duration

`duration` is optional. It resolves in this order, first match wins:

1. `duration` on the phase (JSON above), carried to the utterance
2. `duration` in the scenario's `sound_effects_config`
3. `DEFAULT_BACKGROUND_NOISE_DURATION` in `dataset/generate_speech/generate_audio.py` (8.0s)

Omitting it everywhere is safe — the module default applies.

Note that ElevenLabs caps a single sound effect at 22 seconds. For longer
ambience, use several consecutive silence phases.

## How it flows through the pipeline

The text stage emits a turn whose speaker is the `background_noise` sentinel
instead of prompting an agent, so no LLM call is made for a silence phase.

The speech stage recognises that sentinel and routes the turn to the
sound-effects backend (`elevenlabs` or `stableaudio`, per `sound_effects_config`)
rather than to TTS. The generated segment is resampled to the scene's sampling
rate, so the requested duration holds for every `--tts-model`, not only the one
whose native rate happens to match.

When scene assembly reaches an ambient turn it mixes the audio in directly,
skipping the pyroomacoustics simulation: ambience is non-directional, so giving
it a speaker position in the room would be wrong.

## Failure behaviour

Ambience never fails a scene. If the backend errors, returns nothing, or the
configured model is unknown, the segment falls back to digital silence of the
requested length and generation continues. Check the logs for
`Falling back to a silent ambient segment` to spot it.
