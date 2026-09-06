# Demo page

The demo site for the corpus: audio samples, wake-up timelines, voice cards,
and the naturalness comparison against real recorded meetings.

The page is plain static HTML with no build step, so it publishes from either
host: `.github/workflows/pages.yml` for GitHub Pages, and the `pages` job in
`.gitlab-ci.yml` for GitLab Pages. Both validate the page and serve this same
folder.

## Layout

```
docs/
├── index.html          the whole page — no build step, no framework
├── build_assets.py     encodes audio and writes data/demo.json
├── serve.py            local server (--check validates the page)
├── data/demo.json      transcripts and sample metadata (generated)
└── assets/
    ├── scenes/         full conversations (mp3, 96 kbps mono)
    ├── voices/         character introductions (mp3, as delivered)
    └── naturalness/    ours vs NOTSOFAR pairs (mp3, 96 kbps mono)
```

Everything the page renders comes from `data/demo.json`, which
`build_assets.py` generates. Nothing in `index.html` is hand-edited per
sample — to change what the page shows, change the script's configuration and
re-run it.

## Testing it

`index.html` fetches `data/demo.json` at load, so it has to be served over HTTP
— opening the file directly shows an error instead of the samples.

```sh
python3 docs/serve.py            # http://localhost:8000
python3 docs/serve.py --check    # verify everything, print a report, exit
```

Use `serve.py` rather than `python3 -m http.server`: the latter ignores `Range`
requests, so audio plays but the timeline cannot seek and clicking a turn looks
broken. `serve.py` answers ranges the way GitLab Pages does, so what you see
locally is what deploys.

`--check` is what the CI job runs before publishing. It reports a problem and
exits non-zero if:

- `index.html` or `data/demo.json` is missing or unparseable
- an element the page's script looks up is absent from the markup, which would
  leave that section empty
- any of the 31 audio files 404s or comes back empty
- a range request is not answered with `206` (the timeline could not seek)
- a turn has no timestamp, starts past the end of its audio, or starts before
  the turn before it — any of which breaks the timeline
- the assistant-absent scene contains assistant turns, or the paired scene has
  no detectable wake-up — either would make the comparison say the wrong thing

### What to look at by hand

`--check` cannot judge whether the page reads well. Worth a look after changing
scenes or styles:

- play a featured scene and click a block mid-bar — audio should jump there and
  the matching transcript row should highlight
- the wake block is outlined, and assistant turns are solid, in the right places
- the `no_va` side of the pair shows no orange at all
- reveal on a naturalness row labels both sides, and hides them again
- dark mode: your OS theme, or DevTools → Rendering → emulate
  `prefers-color-scheme`
- narrow the window to phone width; nothing should scroll sideways except the
  pipeline diagram, which scrolls inside its own frame

## Refreshing the samples

Source audio lives outside the repository, under
`~/Databases/intelligent_wakeup`. `build_assets.py` holds the list of scenes the
page shows and where each one comes from; it is the record of that choice.

```sh
python3 docs/build_assets.py --check   # verify sources, encode nothing
python3 docs/build_assets.py           # encode and write data/demo.json
```

The script re-encodes only what changed, validates every ElevenLabs voice id
against `dataset/config/voice_mapping.json`, and fails loudly if a source is
missing. Current output is about 22 MB.

Three constants at the top decide what the page shows:

| constant | what it selects |
| --- | --- |
| `FEATURED` | full scenes with a timeline and transcript |
| `PAIR` | the with/without twin, plus optional trim limits |
| `NATURALNESS_PAIRS` | our meeting ↔ NOTSOFAR meeting, one row each |

Edit those and re-run. Everything below explains what to watch for when you do.

### Turn boundaries

The corpus `time` field is unusable for the timeline: it is planned during text
generation from an assumed 11 characters per second, while the synthesised
speech runs near 19, so those values drift past the end of the audio — more
than half the turns in a typical scene.

Instead the builder transcribes each scene with faster-whisper (`base.en`, a
few seconds per scene on CPU) and matches each turn's opening words to the
recognised word stream. Every turn on the page lands on a real word onset.

The script itself runs on Python 3.9 and later, but the alignment needs
`faster_whisper`, which the project env has:

```sh
/path/to/envs/intelligent_wakeup/bin/python docs/build_assets.py
```

Without it the script falls back to splitting the scene by character count,
which averages ~0.6 s of error, and records `"timing": "estimated"` on the
affected scene instead of `"aligned"`. Check that field if the timeline ever
looks off.

### Build warnings

After encoding a scene the script measures pitch per speaker and warns when a
speaker's voice contradicts their declared gender in `voice_mapping.json`, or
when two people in one scene land within 40 Hz of each other and so will not
read as two speakers:

```
WARNING <scenario>: <name> is male but sounds like 193 Hz
WARNING <scenario>: speakers are only 9 Hz apart, they will sound like one person
```

These are warnings, not errors — the build completes either way, so read them
and pick a different scene if one fires.

### Trimming a scene

Some scenes stay substantive for twenty turns and then circle the same
goodbye to the end. `PAIR` takes optional `with_turns` / `without_turns` to cut
the sample at a given turn, audio and transcript together, with a one-second
fade so it does not end on a hard edge. Remove them to publish the full scene.

### Choosing scenes

Scenes on the page were picked from the v0.4.0 release for an explicit wake-up
followed by a contextual follow-up, no placeholder speaker names (`User1`), no
leaked nested exchanges in a turn, and low filler density. Not every generated
scene meets that bar, so check a candidate before adding it — and listen to it,
since neither the script nor `serve.py --check` can judge whether a scene reads
naturally.

## Publishing

Both pipelines validate `docs/` with `serve.py --check` before publishing, so a
page whose samples are missing fails the run instead of deploying.

### GitHub Pages

`.github/workflows/pages.yml` builds and deploys on every push to the default
branch that touches `docs/`. Enable it once, under **Settings → Pages → Build
and deployment → Source: GitHub Actions**; the URL then appears there and in
the workflow run. `workflow_dispatch` lets you publish without a push.

### GitLab Pages

The `pages` job in `.gitlab-ci.yml` does the same on the default branch, and
`pages:preview` is a manual job on merge requests that publishes to a
per-branch sub-path so it cannot overwrite the live site. Both print the
address as `Page URL` in the job log.

Pages has to be configured on the instance for either URL to resolve — where it
is not, the job still leaves `public/` as a downloadable artifact, which is the
way to see the page.

To verify what the job will publish, run its steps locally:

```sh
rm -rf public && mkdir public && cp -r docs/. public/
rm -f public/iw-text_gen_pipeline-*.pdf
```

## Pending links

The masthead has a disabled `Paper` placeholder and a commented-out `Dataset`
button. When the preprint and the dataset download exist, swap them in — both
are marked with comments in `index.html`.
