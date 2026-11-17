# tsutawaru

**伝わる** — *for meaning to get through.* An intransitive verb: nobody performs it, the meaning simply arrives. It runs in both directions, which is the point — understanding what is said to you, and eventually being understood yourself.

Open-source, native, low-latency live Japanese → English audio translation with Hepburn romaji and per-token morphological breakdown, designed for Discord voice calls, streams, and any system audio source.

Built to follow Japanese friends in Discord voice calls in real time, and to learn the language while doing so — which is why the romaji and per-token gloss tiers are core rather than decorative. A pure captioning tool would show you the English and stop. The end goal is to stop needing this one.

> Previously named `kotoba-live` (renamed 2026-08-18). `kotoba-tech/kotoba-whisper` is an established Japanese ASR model family — one this project can now load with `--asr kotoba` — so the old name collided with a dependency inside its own CLI. Any `kotoba` still in this repository refers to that model, never to the project.

## Key Features

- **Decoupled Low-Latency Pipeline:** Real-time VAD boundary detection, local speech recognition (Qwen3-ASR by default, Whisper models selectable), token-level morphological analysis, and multi-threaded sentence/gloss translation.
- **Offline by Default:** Speech recognition, sentence translation (Qwen3.5-4B on MLX) and the word breakdown (JMdict) all run on the Mac. A call can run with the network unplugged.
- **Learner-First Morphological Breakdown:** Intelligently agglutinates bound morphemes (e.g. `おはようございます`, `ですね`), pairs each word with dictionary form and romanization, and resolves particles at zero latency.
- **Per-Application Capture:** On macOS, taps one application directly through Core Audio — so a game running next to a voice call is never captured in the first place. No virtual audio driver, no output routing.
- **Silent & Non-Interfering:** Read-only capture. No TTS, no audio playback, ever.
- **Readable Live Window:** Each line is a card; the newest one is highlighted as the active line. Japanese appears as a provisional line while someone is still talking and is replaced in place when they finish. Text is copyable, English is set in a Latin face, and the text column stops at a readable width however wide the window gets.
- **Multiple Output Sinks:** The window above, a rich terminal live view, a floating transparent desktop overlay (PyQt6), and a WebSocket server for OBS Browser Docks and browser overlays.

## Status

*As of 2026-11-17. Developed and measured on macOS / Apple Silicon (M5 Air, 16 GB).*

| Mode | State |
|---|---|
| YouTube / streams | In regular use: eight live sessions on the current pipeline since 2026-09-29. |
| Discord calls | Capture proven on a real call (2026-08-22), on an older pipeline. Not yet run live on the current one. |
| Windows / Linux | Code paths exist; untested. The default models need Apple Silicon. |

**Measured on one recorded Discord call** (600 utterances, 23 minutes), run through the current pipeline:

- **Speech recognition:** p50 205 ms per utterance. On the 11 hardest lines, graded by ear, the Japanese was right on 10. The 11th was unintelligible to a human listener too.
- **Translation:** 600 of 600 lines translated offline by the default model, PLaMo 3 Translate 2B, at p50 258 ms / p95 649 ms. One line came out objectively broken (romanized Japanese instead of English). Scored against a draft answer key for 335 of those lines, it is well ahead of the model it replaced on 2026-11-17 (see [Translation](#translation)).
- **End to end:** p50 754 ms / p95 1877 ms from the voice detector closing a line to its English being ready, queue waits included, replayed in real time with nothing dropped. The detector waits about 450 ms after speech stops before it closes the line, so add that to get the delay behind the speaker. Measured on 2026-10-08 with the previous translator, which took about 65 ms longer per line; not re-measured since.

**Known weaknesses**, in the order they are being worked on:

1. **Short fragments.** Japanese often leaves out who is speaking. On a fragment, the translator completes a full English sentence and fills in a subject, usually "I", that the speaker never said. On those same 11 hard lines the English was wrong on 8 (a sample picked because it looked suspicious, so not an error rate). Against the draft call key, the current translator adds an "I", "we" or "you" the key does not have on 56 of 298 lines; the previous one did on 69.
2. **A weak doubt signal.** Lines the speech recognition was unsure of are dimmed and tagged "unsure" (about 7% of call lines), and the least certain 1% are dropped. On the recorded six-person call about one line in five is garbled, and the dimming catches about one in six of those: it separates well on long lines and hardly at all on lines of a few characters. Not yet checked by watching it live.
3. **Very loud game audio.** On the loudest stream tested, the voice detector misses about 20% of speech starts.

## Quick Start

### 1. Requirements & Setup

- Python 3.14 (or 3.11 fallback)
- Apple Silicon for the defaults. Qwen3-ASR and the MLX translator (PLaMo 3 Translate 2B, 2.3 GB) download on first run.
  On other platforms pass a Whisper model and a network translator, e.g.
  `--asr kotoba --translator google`.

```bash
# macOS (Apple Silicon)
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-mac.txt

# Windows / Linux (CUDA)
pip install -r requirements.txt -r requirements-cuda.txt
```

Build the offline gloss dictionary once (JMdict, ~41 MB, CC BY-SA 4.0). Without it
the breakdown tier falls back to translating each word over the network:

```bash
python -m tsutawaru.translate.jmdict
```

On **macOS 14.4+** nothing further is needed — capture defaults to a per-application tap. Grant **Screen Recording** and **Microphone** to *ProcTap Helper* under System Settings → Privacy & Security the first time you run it.

On **Windows and Linux**, capture falls back to a virtual loopback device, which needs a driver:

- **Windows:** VB-CABLE (`https://vb-audio.com/Cable/`) or PyAudioWPatch loopback
- **macOS (if you opt out of tapping):** BlackHole 2ch + Multi-Output Device (`brew install --cask blackhole-2ch`)

### 2. Check Capture

```bash
# Per-process: which apps can be tapped, and which pid carries their audio
python run.py --list-sources
python run.py --source-test discord --seconds 5

# Device path: list devices, then test levels on your loopback device
python run.py --list-devices
python run.py --audio-test --device "BlackHole" --seconds 5
```

### 3. Run Live Translation

```bash
# macOS: tap Discord (per-process capture is the default)
python run.py --source discord

# ... or a stream playing in Chrome
python run.py --source youtube

# Pick the models yourself (defaults: --asr qwen3 --translator plamo3)
python run.py --source discord --asr kotoba --translator google

# Force the device path, or run on Windows/Linux
python run.py --audio-backend device --device "BlackHole"

# Floating overlay + WebSocket server
python run.py --sink both --ws-port 8765

# Show rolling latency metrics on exit
python run.py --stats
```

### 4. Keep the Session

```bash
# Timestamped JSONL of every line, all four tiers, into experiments/sessions/
python run.py --source discord --record

# ... or choose the path yourself
python run.py --source discord --record-to logs/call.jsonl

# Markdown study sheet: frequency-ranked vocabulary + the full transcript
python run.py --source discord --export-md study.md
```

Both are written when the session shuts down, not as lines arrive, because a
segment is still being filled in after it first appears — the word-gloss lane
finishes before the sentence translation lands. The JSONL is additionally
rewritten every 25 lines so an unclean kill costs at most a few entries.

The study sheet groups inflected forms under their dictionary form (潜って and
潜った both count toward 潜る) while listing the surfaces actually heard, since
recognising the inflected form is the skill being practised. Its glosses are
dictionary senses looked up on the base form, so read them against the
transcript rather than on their own.

## Translation

| Tier | Default | Alternatives |
|---|---|---|
| Sentence | PLaMo 3 Translate 2B on MLX, offline (`--translator plamo3`) | `plamo3-small`, `qwen3.5` (offline), `google`, `deepl` (network), `none` |
| Breakdown | JMdict, then the sentence provider for what it lacks | — |

The offline translator needs `mlx-lm` and Apple Silicon. The default is
[PLaMo 3 Translate](https://huggingface.co/pfnet/plamo-3-translate-2b-MLX-4bit)
2B, a translation-only model from Preferred Networks, downloaded on first run.
JMdict answers 96.7% of breakdown tokens, so nothing in a call needs the
network.

It became the default on 2026-11-17, replacing Qwen3.5-4B, on these
measurements. Both models were run through the same code on the same lines:

| | PLaMo 3 Translate 2B | Qwen3.5-4B |
|---|---|---|
| BSD business dialogue, 1,052 lines with human English (chrF) | **43.5** | 39.2 |
| Recorded call, 335 lines against a draft key (chrF) | **46.7** | 39.1 |
| Call lines given an "I", "we" or "you" the key lacks (of 298) | **56** | 69 |
| Call statements turned into questions (of 292) | **19** | 82 |
| Call questions kept as questions (of 43) | 32 | **39** |
| Stream questions the speech recognition flattened, kept (of 124) | 54 | **63** |
| Per call line, p50 / p95 | **258 / 649 ms** | 322 / 777 ms |

The gap on the call lines is 7.6 chrF, with a 95% interval of 5.1 to 10.2. The
call key is a draft: it was written from the transcript, not by ear, and covers
one call. On the 28 lines that were graded by ear it agrees with the listener
(lines judged wrong average 31.4 against it, lines judged fine 57.1).

Live is slower than those offline figures. In its first live stream session
(264 lines, 2026-11-17) a translation took p50 504 ms / p95 1467 ms, because
the translator shares the GPU with speech recognition and is given context on
a stream. Speech to English was p50 919 ms / p95 2928 ms, about the same as
the sessions before the change. No line was left untranslated and none of the
translator's guards fired.

PLaMo takes no instructions, so names go in through its glossary, and tsutawaru
puts back the "？" the speech recognition drops before translating. That is also
where it is weaker: it keeps fewer questions than Qwen3.5, which can be told to.
Its tokenizer is code that ships with the model, so tsutawaru runs it only at
the two model commits it pins. `plamo3-small` is the 0.8B (BSD chrF 43.1, p50
91 ms). `--translator qwen3.5` brings the previous default back. Three other
translation models were measured on the same lines and were not better: Hy-MT2
1.8B and 7B, and TranslateGemma 4B. Built with PLaMo.

A local translator has been the default since 2026-09-29, when Google's free
endpoint started refusing this machine with a 429 CAPTCHA and a network default
shipped an app that could not translate.

On a stream, the translator is given the previous three lines as context,
because a stream is one speaker. On a call it is not: tsutawaru cannot tell
speakers apart. Measured on the recorded call, context made Qwen3.5 worse and
gave PLaMo no measurable gain.

## Audio Capture

`[audio] backend` selects how audio is captured, and defaults to `"auto"`:

| Value | Behaviour |
|---|---|
| `"auto"` | `"process"` on macOS 14.4+ with `proc-tap` and `psutil` installed, `"device"` otherwise. Falls back quietly and logs which it chose. |
| `"process"` | Per-application Core Audio tap. Never falls back — if it cannot run, startup fails and names the reason. |
| `"device"` | Captures an input device (virtual loopback). Records the whole system mix. |

Per-process capture is the default because **signal-to-noise is what decides whether a line transcribes correctly**, and it is the only part of the pipeline that improves the signal rather than coping with a degraded one. A game and a voice call fused into one system mix cannot be separated by any downstream model; tapping one application means the game is never in the recording.

It does not help when the interference is already mixed *upstream* — a streamer's game audio arrives inside their broadcast, so tapping the browser captures both.

### Which process carries an app's audio

A Core Audio tap binds to exactly one pid, and being Chromium-based does not tell you which. Measured here:

| App | Audio comes from | Its audio service delivers |
|---|---|---|
| Google Chrome | `audio.mojom.AudioService` | works |
| Discord | its **renderer** | 0 bytes |

Discord's voice engine is a native module that opens its own Core Audio stream instead of going through Chromium's audio service, so the obvious rule ("tap the audio service") captures silence from Discord forever while working perfectly for Chrome.

This is not discoverable at runtime — an unentitled process cannot enumerate audio objects, and a tap on the wrong pid **opens successfully and then delivers nothing**, identical to an idle app. So each source declares a preference order and capture **escalates** to the next candidate after 12 s of silence, wrapping around. A wrong guess costs seconds, not the feature. `--list-sources` prints the chosen pid and its fallbacks so an escalation in the log is legible.

### When a tap is silent

Silence has four causes and they look alike. `--source-test` separates them; this is the ranking it uses.

1. **The app isn't playing.** Chromium apps only own a Core Audio process object while audio is actually flowing, and retire it when idle. Normal — capture resumes on its own.
2. **The app's output device doesn't match your system default.** ProcTap wraps the tap in an aggregate device whose sole subdevice is the *current default output*. An app pointed anywhere else is outside that graph and captures as pure silence **while remaining perfectly audible**. Browsers follow the system default and rarely hit this; Discord, Zoom and OBS ship their own output picker and hit it constantly. Set the app's output to your system default.
3. **Screen Recording is not granted to *ProcTap Helper*.** Core Audio creates the tap successfully without that consent and then feeds it silence, with no error at any layer. The signature is packets arriving with every sample zero.
4. **Wrong pid.** Escalation handles this automatically; see above.

### Limits

- macOS 14.4+ only. Everywhere else `"auto"` resolves to the device path.
- **Per app, not per tab** — tapping Chrome captures every tab, so a second noisy tab is included.
- **One tap at a time.** ProcTap's helper is a singleton `.app`, so a second concurrent tap fails with LaunchServices `-1712`. Two tsutawaru instances cannot both capture, and an external probe fails while tsutawaru holds the tap. This is also deliberate downstream: two sources would mean two concurrent Whisper inferences, which on MPS is a thread-safety abort this project has already hit once.
- ProcTap's own `--list-audio-procs` is heuristic and lists processes that cannot play audio. `--list-sources` uses the audio-service check instead and is accurate.

## Repository Layout

```
run.py                  CLI entrypoint
config.example.toml     every setting, with defaults (copy to config.toml)
tsutawaru/              the package
  audio/                capture (per-process tap, device, WASAPI), VAD, resampling
  stt/                  Whisper / kotoba / Qwen3 engines behind one factory
  nlp/                  tokenize -> filter -> agglutinate -> romaji
  translate/            sentence translators, JMdict glosses, cache, worker pool
  pipeline/             orchestrator, queues, session recorder
  ui/                   Qt window and overlay, console sink, WebSocket server
tests/                  pytest suite (no network, no models)
tools/                  standalone analysis scripts, not imported by the app
  grade.py              play clips and grade a sheet by ear (--rows to re-check)
  defects.py            reference-free defect counter for JP->EN lines
experiments/
  sessions/             recorded sessions; `--record` writes here by default
  reports/              measurement and comparison reports
```

`experiments/` is gitignored and stays on your machine: the recordings
themselves (WAV, JSONL, console logs) and the sheets generated from them
(`--export-md` study sheets, `compare4.py` reports). `tools/grade.py` plays
clips from the `wav/` directory next to the sheet it is grading, so a sheet and
its audio stay in one folder.

## License

GPL-3.0, because `pykakasi` and `PyQt6` are GPL. Both have permissive
substitutes (`cutlet` for romaji, a Tk or web overlay for the window), so a
build without them can be MIT.

The default translator, PLaMo 3 Translate, is not part of this repository: it
downloads from Preferred Networks under the
[PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-en),
which is free for personal and academic use and needs registration for
commercial use. `--translator qwen3.5` uses an Apache-2.0 model instead.
