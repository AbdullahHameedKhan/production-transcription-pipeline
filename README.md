# Transcription Pipeline

Converts audio into timestamped text. Handles any format ffmpeg can decode, handles files of any length, and exposes both a CLI and an HTTP API.

Built for the Volga Partners Software Engineer assessment.

---

## What it does

```
audio file (any format, any length)
      |
      v
   PROBE        ffprobe: what is this file, how long is it
      |
      v
   NORMALIZE    ffmpeg: convert everything to 16 kHz mono PCM
      |
      v
   CHUNK        cut long files into overlapping 5 minute windows
      |
      v
   ASR          faster-whisper transcribes each window
      |
      v
   STITCH       shift timestamps onto the source timeline, drop duplicates
      |
      v
   EXPORT       json | srt | vtt | text
```

Each stage is independent. Probe does not convert, normalize does not cut, stitch never touches a file. That separation is what makes each piece testable and replaceable in isolation.

---

## Requirements

- Python 3.11 or newer
- ffmpeg, including ffprobe (system binary, not a pip package)

### Installing ffmpeg

**Windows**
```powershell
winget install Gyan.FFmpeg
```

**macOS**
```bash
brew install ffmpeg
```

**Debian / Ubuntu**
```bash
sudo apt install ffmpeg
```

Verify both binaries are on PATH:
```bash
ffmpeg -version
ffprobe -version
```

---

## Setup

```bash
git clone https://github.com/AbdullahHameedKhan/production-transcription-pipeline.git
cd production-transcription-pipeline

python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\activate

pip install -e ".[whisper,api,dev]"
```

### What the extras contain

| Extra | Contents | Needed for |
|---|---|---|
| base | pydantic-settings, typer | Core pipeline and CLI parsing |
| `whisper` | faster-whisper | Actually transcribing |
| `api` | fastapi, uvicorn, python-multipart | The HTTP service |
| `dev` | pytest, httpx, ruff, mypy | Tests and linting |

The ASR engine is an optional extra on purpose. A plain `pip install -e .` gives you the full pipeline, exporters and their tests without pulling a large model runtime, because everything depends on the `ASREngine` protocol rather than on faster-whisper directly.

---

## Usage

### Command line

```bash
transcribe audio.mp3                              # JSON to stdout
transcribe audio.mp3 --format srt                 # SRT subtitles
transcribe audio.mp3 --format vtt -o out.vtt      # WebVTT to a file
transcribe audio.mp3 --format text                # plain text only
transcribe audio.mp3 --model tiny                 # faster, less accurate
transcribe audio.mp3 --language en                # skip auto detection
```

Equivalent to `python -m transcription.cli`.

First run downloads the Whisper model (roughly 140 MB for `base`) and caches it.

**Example output**

```
1
00:00:00,910 --> 00:00:15,340
Hello, this is testing. My name is Abdullah. I am from Islamabad.

2
00:00:15,340 --> 00:00:27,060
I have completed my masters in artificial intelligence.
```

### HTTP API

```bash
uvicorn transcription.api.main:app --reload
```

Interactive docs at `http://127.0.0.1:8000/docs`.

Transcription can take minutes, so the API is asynchronous. You submit, receive a job id immediately, then poll.

```bash
# 1. submit
curl -X POST -F "file=@audio.mp3" http://127.0.0.1:8000/v1/transcriptions
# -> {"job_id":"a1b2c3...","status":"queued","status_url":"/v1/transcriptions/a1b2c3..."}

# 2. poll until completed
curl http://127.0.0.1:8000/v1/transcriptions/a1b2c3...
# -> {"job_id":"a1b2c3...","status":"running","attempts":1,...}

# 3. fetch in any format
curl "http://127.0.0.1:8000/v1/transcriptions/a1b2c3.../result?format=srt"
```

### Endpoints

| Method | Path | Returns |
|---|---|---|
| `GET` | `/v1/health` | Service status, loaded engine and model, active job count |
| `POST` | `/v1/transcriptions` | `202` with a job id and status URL |
| `GET` | `/v1/transcriptions/{id}` | Job state, attempt count, error if any |
| `GET` | `/v1/transcriptions/{id}/result?format=` | Transcript as json, srt, vtt or text |

### Status codes

| Code | Meaning |
|---|---|
| `202` | Upload accepted, processing |
| `400` | Unsupported export format, or empty upload |
| `404` | Unknown job id |
| `409` | Result requested before the job completed, or the job failed |
| `413` | Upload exceeds the size limit |
| `415` | Unsupported file extension |

`409` rather than `404` for an unfinished job is deliberate: the resource exists and the request is valid, but the current state does not permit it.

---

## Docker

Removes the ffmpeg system dependency entirely.

```bash
docker build -t transcription-pipeline .
docker run -p 8000:8000 transcription-pipeline
```

---

## Configuration

Every tunable value lives in `config.py` and can be overridden by an environment variable or a `.env` file, with no code change.

```bash
TRANSCRIPTION_WHISPER_MODEL=small
TRANSCRIPTION_CHUNK_LENGTH_S=600
TRANSCRIPTION_MAX_CONCURRENT_JOBS=4
```

| Setting | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL` | `base` | tiny, base, small, medium, large-v3 |
| `WHISPER_DEVICE` | `cpu` | cpu or cuda |
| `WHISPER_COMPUTE_TYPE` | `int8` | int8, float16, float32 |
| `TARGET_SAMPLE_RATE` | `16000` | Normalization target |
| `TARGET_CHANNELS` | `1` | Mono |
| `CHUNK_LENGTH_S` | `300` | Window size for long files |
| `CHUNK_OVERLAP_S` | `5` | Overlap between windows |
| `MAX_CONCURRENT_JOBS` | `2` | Simultaneous transcriptions |
| `MAX_ATTEMPTS` | `3` | Retries for transient failures |
| `MAX_UPLOAD_BYTES` | `209715200` | 200 MB upload cap |

All keys take the `TRANSCRIPTION_` prefix.

---

## Testing

```bash
pytest -v
```

28 tests, under a second, with no audio files and no model downloads.

```bash
# load test against a running server
python tests/test_load.py --file audio.ogg --users 5
python tests/test_load.py --file audio.ogg --users 10 --rounds 2
```

### What is tested and what is not

The ffmpeg wrappers have no unit tests. If ffmpeg fails it fails loudly with a non zero exit code and an error message, so there is no subtle wrong answer to catch.

The reassembly arithmetic is tested exhaustively, because it can produce output that looks entirely correct but is wrong: timestamps drifted by minutes, words silently missing at every seam, words duplicated at every seam. None of that raises an error, and none of it appears on a short test file, since a file under five minutes never gets chunked at all.

`test_stitch.py` asserts that windows always cover the full timeline with no gaps, that overlapping words survive exactly once, that a word past a seam transfers to the correct owner rather than vanishing, and that chunk local times are lifted correctly onto the source timeline.

---

## Design decisions

### Formats are normalized, not handled

Every input is converted through ffmpeg into one canonical form before anything else touches it: **16 kHz, mono, uncompressed 16 bit PCM**. That applies to wav input too, since a wav might still be 44.1 kHz stereo.

16 kHz because that is what Whisper is trained on, so anything else gets resampled internally anyway. Mono because speech gains nothing from stereo. Uncompressed PCM because the audio is already decoded and PCM allows exact sample level seeking when cutting chunks.

The alternative is a branch per format, which grows forever and pushes format knowledge into every downstream function. After normalization, no code in this system asks what the input format was.

The cost is a dependency on ffmpeg as a system binary, which the Dockerfile pins.

### Long files use overlapping windows

Files longer than five minutes are cut into 300 second windows that overlap by 5 seconds.

The overlap exists because cutting at exactly 300 seconds may slice through the middle of a word, destroying it in both pieces. With overlap, anything damaged at one window's edge sits comfortably inside the next.

Chunking bounds memory regardless of input length and limits the cost of a failure to a single window rather than the whole file.

### Reassembly resolves the duplicates overlap creates

Each chunk is transcribed as an isolated file, so the model reports times starting from zero. Every segment is shifted by its chunk's absolute start position onto the source timeline.

The overlap means the shared region is transcribed twice. Ownership is resolved at the midpoint of each overlap: **a segment belongs to whichever chunk's territory its centre falls in.**

For chunks covering 0 to 300 and 295 to 595, the seam is at 297.5. A word at 296 to 297 has a centre of 296.5, so chunk 0's copy is kept and chunk 1's is discarded. Exactly one copy survives, with no gap and no duplicate.

The centre is used rather than the start because the model does not segment identically when given different audio boundaries. The centre sits in the middle of actual speech and is stable across those differences; an edge is wherever the model happened to cut.

### The ASR engine sits behind a protocol

```python
class ASREngine(Protocol):
    def transcribe(self, audio_path: Path, *, language: str | None = None) -> ASRResult: ...
```

The pipeline never imports faster-whisper. It depends only on this contract.

Adding Groq, Deepgram or AssemblyAI means writing one class and adding one line to `build_engine`, with no changes to the pipeline. It also means tests can inject a five line fake engine in place of a large model, which is why the suite runs in under a second.

faster-whisper was chosen as the default because it runs locally with no API key or network, is roughly four times faster than reference Whisper through the CTranslate2 runtime, and performs acceptably on CPU with int8 quantisation.

### Accepting work and executing work are separate

Uploads are cheap disk IO. Transcriptions spawn ffmpeg and occupy a model.

Every upload is accepted immediately and streamed to disk in one megabyte pieces, so memory stays flat regardless of file size, and the size cap is enforced during the transfer rather than after it. A semaphore then caps how many transcriptions run simultaneously. Ten uploads all succeed and queue; two execute at a time.

Transcription runs through `asyncio.to_thread`, because it is blocking CPU work that would otherwise freeze the event loop and stall every other request including health checks.

### Retries distinguish permanent from transient

Bad input raises `ValueError` and is never retried, since replaying it produces the identical error. The job fails immediately with a clear message.

Everything else is treated as transient and retried up to three times with exponential backoff of 1, 2 then 4 seconds, so a system already under pressure is not hammered further.

Attempt count is stored on the job and exposed to clients, so a caller can distinguish a job that failed instantly from one that fought through three attempts.

### Timestamps are segment level

Word level timestamps are available from the model and would be a config flag. Segment level was chosen because it is what subtitles, search and playback actually consume, and because it reassembles more cleanly across chunk boundaries.

---

## Known limitations

**Jobs do not survive a restart.** The job store is in memory. This is deliberate rather than an oversight: it sits behind a four method `JobStore` protocol, so replacing it with Redis or Postgres means writing one class and changing nothing else. In production the in process worker also becomes a real queue with separate worker machines.

**Fixed windows rather than voice activity detection.** VAD based splitting would cut at natural silences and avoid the duplicate problem entirely. Fixed windows were chosen because they are deterministic and the reassembly arithmetic can be proven correct with unit tests, whereas VAD adds a tuning problem that varies by audio type. The seam ownership rule also assumes segments are much shorter than chunks, which holds at the default 300 second windows against roughly 15 second segments but degrades if chunks are shrunk toward segment length.

**Audio is not retained.** Uploads stream to a working directory and are deleted once transcribed. Production would store audio in object storage with presigned upload URLs so bytes never pass through the API, and transcripts in Postgres where they can be queried.

---

## Production notes

| Concern | Current | Production |
|---|---|---|
| Job store | In memory dict behind a protocol | Redis or Postgres, same four methods |
| Workers | `BackgroundTasks` in process | Celery, SQS or similar with separate workers |
| Audio storage | Temp directory, deleted after use | S3 with presigned upload URLs and a lifecycle policy |
| Transcript storage | In memory with the job | Postgres, segments as JSONB or a related table |
| Duplicate submissions | Not deduplicated | Idempotency keys on submission |
| Exhausted retries | Marked failed | Dead letter queue for inspection |
| Client notification | Polling | Webhooks |
| Scaling | Single process | Stateless pipeline scales horizontally; workers on GPU nodes |

---

## Project structure

```
src/transcription/
├── config.py           every tunable value, env overridable
├── models.py           shared data types, no dependencies
├── asr.py              ASREngine protocol + faster-whisper implementation
├── stitch.py           timestamp shifting and overlap resolution
├── pipeline.py         orchestrates the stages, no logic of its own
├── exporters.py        json, srt, vtt, text
├── cli.py              command line entry point
├── audio/
│   ├── probe.py        ffprobe inspection, never trusts extensions
│   ├── normalize.py    ffmpeg conversion to the canonical format
│   └── chunk.py        overlapping window planning and cutting
└── api/
    ├── schemas.py      public JSON contract, separate from models.py
    ├── jobs.py         job store, worker, concurrency, retries
    ├── endpoints.py    HTTP routes only
    └── main.py         app factory, loads the model once at startup

tests/
├── test_stitch.py      the reassembly arithmetic
├── test_exporters.py   output format contracts
└── test_load.py        concurrency under load, run manually
```

Roughly 900 lines total.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'faster_whisper'`**
The ASR engine is an optional extra. Run `pip install -e ".[whisper]"`.

**`ffmpeg not found on PATH`**
Install ffmpeg for your platform (see Requirements) and restart your shell so the PATH change takes effect.

**`pytest: command not found`**
Run `pip install -e ".[dev]"`.

**First transcription is slow**
The Whisper model downloads on first use and is then cached. Use `--model tiny` for a faster loop during development.
