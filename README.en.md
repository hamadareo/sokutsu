# Sokutsu (即通, "soku-tsū")

[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![platform: macOS Apple silicon](https://img.shields.io/badge/platform-macOS%20(Apple%20silicon)-lightgrey.svg)](#requirements)

[日本語](README.md) | English

<img src="src/tsuyaku/webui/logo.svg" width="72" alt="Sokutsu">

**Sokutsu is a real-time, two-way interpreter for face-to-face conversations around a single Mac. It shows large translated captions and speaks the translation aloud, and it turns what was said into meeting notes.** Speech recognition, translation and text-to-speech can all run on your own Mac (no cloud API is used), so audio and text need not leave the machine.

- Languages: Japanese, English, Chinese, Korean (any pair of two; other languages are shown as captions only)
- Two ways to run: **"This Mac"** (fully local) or **"Remote GPU"** (optionally offload translation to your own inference server)

> No screenshots are included because they would show real utterances. See "Usage" for a description of the screen.

The Japanese README is the primary one and is more detailed; this page is a summary.

## Features

- Captions first: big translated caption + original text + a conversation history rail. A dotted-underline *predicted* translation appears while the speaker is still talking and is finalized when they stop.
- Spoken translation with per-language voice and speed; an automatic pace mode adapts speaking speed to the speaker. The mic pauses while the app speaks (use "captions only" for online meetings).
- Automatic language detection; built-in glossary (ja/en/zh/ko) plus your own term dictionary to fix misheard acronyms and names.
- Meeting notes with templates (regular meeting, 1:1, sales, interview, lecture, brainstorm, Q&A): summary, decisions, action items, speaker names, all with a local LLM. Ask questions about one or all meetings, with source lines cited.
- Library with folders, tags, stars and search. Export to PDF, Word (.docx), Markdown, and subtitles (SRT / WebVTT). Import audio/video files (m4a, mp3, wav, mp4, mov, ...) to get transcripts, speakers and notes.
- Speaker separation and "remember voices" (voice features stay on this Mac). Optionally hear the other side of an online meeting ("Mic + this Mac's sound", macOS 14.2+).
- Calendar / Reminders integration only when you press the buttons. UI in Japanese, English, Chinese, Korean.

## Requirements

- Apple silicon Mac (M1 or later), macOS 13.3 or later (14.2+ for capturing the other side of online meetings)
- 16 GB memory recommended (8 GB works, slower); about 5 GB free disk (models are about 3.4 GB)
- From source: Python 3.10+ (tested on 3.10) and clang (`xcode-select --install`, used to build a small system-audio helper; optional)
- A translation engine: for "This Mac" when running from source, an OpenAI-compatible server on port 1234 (e.g. [LM Studio](https://lmstudio.ai) with a Gemma 3 4B model). A DMG you build yourself bundles llama.cpp's `llama-server` + Gemma 3 4B instead.

## Getting started

There is no prebuilt app yet. Run from source or build your own DMG.

```sh
git clone https://github.com/hamadareo/sokutsu.git
cd sokutsu
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/get_models.sh   # speech / VAD / TTS models from their official sources (first time only)
./run.sh                     # start your OpenAI-compatible server on port 1234 first
```

- Optional Japanese voice: [VOICEVOX ENGINE](https://voicevox.hiroshiba.jp/) unpacked in `~/voicevox` (override with `VOICEVOX_HOME`); otherwise macOS `say` is used.
- **Remote GPU (optional)**: run an OpenAI-compatible LLM server on a machine you control, make it reachable over key-based SSH, then `TSUYAKU_SSH=user@host ./run.sh`. `scripts/remote_tunnel.sh` sets up the SSH port forward; pick "Remote GPU" or "Auto" on the start screen. Optionally `server/remote_stt_server.py` (faster-whisper) can also run speech recognition remotely; in that case the audio being translated is sent to that server over SSH.
- Build a DMG: `bash scripts/build_dist.sh` (models are downloaded on first launch from official sources and verified by SHA256).

## Usage

Pick a language pair (optionally topic / names / terms, a term dictionary, a meeting type) on the start screen and press "Start". Main keys: `Space` or click a caption = stop speaking and listen again; `Esc` = cancel / quit; `-` `+` = caption size; `V` = voice on/off; `B` then `1`-`4` = mark the current line (important / decision / to-do / question); `M` = write a note; `?` = cheat sheet. Right-click rows and items in meeting views to fix text, change speakers, mark or delete (`⌘Z` to undo).

## How it works

```
mic / system audio -> auto gain -> Silero VAD -> ear (SenseVoice / Whisper / optional remote faster-whisper)
   -> term fixing (glossary) -> translation LLM (predicted -> final) -> captions + TTS (Kokoro / VOICEVOX / macOS say)
   \-> local log -> meeting notes / Q&A / export
```

## Measured results

Measured during development, mostly on an M3 / 16 GB Mac. See [eval/README.md](eval/README.md) for methods. Treat as rough guides.

| | This Mac (Gemma 3 4B) | Remote GPU (Qwen-family model) |
|---|---|---|
| chrF, FLEURS 30 sentences x 12 directions | 39.03 / 39.34 (two runs) | 45.03 / 45.06 (two runs) |
| Terms present in translation (74 items x 12 directions = 888) | 93.6% | 98.3% |

COMET (wmt22-comet-da) of the bundled Gemma 3 4B QAT Q4_0 with the app's prompt, over 1,106 translations (4 sets): mean 87.83 (87.43 before the length-retry and prompt-example changes).

Latency on This Mac, Japanese to English, fixed speed, 18 made-up sentence pairs played in real time (median / p90 from the end of speech, LM Studio, build of 11/08): translation final 0.85 / 1.69 s, voice onset 1.56 / 2.55 s. In fast back-and-forth speech the remote setup skipped 4 to 7 of 18 spoken translations (captions still appeared).

Real English meetings (AMI corpus, 8 clips x 3 min, remote setup, real time): final-caption delay median 4.93 s / p90 10.24 s, WER 38.5%; 57.7% of reference words appeared in final captions. Overlapping speech is still weak.

## Privacy

In "This Mac" mode audio, text and translations stay on the Mac; the network is used only to download models from their official sources. No cloud API is used. Only if you choose "Remote GPU" do translated text (and audio, if you use the remote ear) go to the server you specify over SSH. Logs, settings, glossary, recordings and voice features are stored locally and excluded from Git.

## Limitations

Designed for face-to-face turn-taking; overlapping speech is weak. Quality depends on the chosen LLM (4B-class model is weaker than a larger remote model on terms and long sentences). Apple silicon macOS only. No prebuilt app yet; running from source needs Python and a separate LLM server. Korean speech output may fall back to macOS voices.

## Development

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

`main` is kept working; changes go through feature branches and are compared before/after on the same evaluation.

## License and acknowledgements

MIT ([LICENSE](LICENSE), "Sokutsu contributors"). Models and third-party software have their own licenses: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) (Gemma: Gemma Terms of Use; SenseVoice: FunASR Model Open Source License). Thanks to whisper.cpp / Whisper, llama.cpp, sherpa-onnx, SenseVoice, Kokoro, Silero VAD, 3D-Speaker, VOICEVOX, Gemma, FLEURS, the AMI corpus, pywebview and many other open-source authors.

## Roadmap

Prebuilt signed app; stronger handling of overlapping speech and far-field mics; a mode that plays each person's translation on earphones; more engine choices configurable from the UI; more languages; broader evaluation sets.
