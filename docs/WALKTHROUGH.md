# Project Walkthrough

Read top to bottom. Each file gets: the problem it solves, the decision I made, and the code.

---

## The problem, in one page

Someone gives you an audio file. You need timestamped text out of it.

Four things make this hard:

1. **Formats are chaotic.** mp3, ogg, flac, different sample rates, embedded album art, wrong file extensions.
2. **Models have limits.** You cannot feed a two hour file to a speech model. It runs out of memory and a crash loses everything.
3. **Cutting breaks timestamps.** Split a file into pieces and each piece thinks it starts at zero. Piece three is wrong by however long pieces one and two were.
4. **It is slow.** A ten minute file takes a minute. HTTP requests cannot wait that long.

The whole design follows from these four.

## The shape of the solution

```
audio file
    |
    v
 PROBE       what is this file really?
    |
    v
 NORMALIZE   convert everything to one format
    |
    v
 CHUNK       cut long files into overlapping windows
    |
    v
 ASR         speech to text, per window
    |
    v
 STITCH      fix the timestamps, remove duplicates
    |
    v
 EXPORT      json, srt, vtt, text
```

Problems 1 and 2 are solved by probe and normalize. Problem 3 by chunk. Problem 4 by stitch. The API solves the slowness.

**One rule holds it together: each step knows nothing about the others.** Probe does not convert. Normalize does not cut. Stitch never touches a file, it only does maths on numbers. That is why each piece can be tested and replaced alone.

---

# THE ROOT FILES

These are the pieces the rest of the code is built from.

## `config.py`

**Problem:** if numbers are scattered through the code, tuning anything means hunting for them. Chunk length hidden in one file, sample rate in another, retry count in a third.

**Decision:** one class, every tunable number, nothing hardcoded anywhere else.

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRANSCRIPTION_", env_file=".env")

    chunk_length_s: float = 300.0      # 5 minute windows
    chunk_overlap_s: float = 5.0       # so no word gets cut in half
    target_sample_rate: int = 16_000   # what Whisper is trained on
    target_channels: int = 1           # speech is mono
    whisper_model: str = "base"
    max_concurrent_jobs: int = 2
    max_attempts: int = 3
```

**The bonus:** inheriting from `BaseSettings` means every value can be overridden by an environment variable. Set `TRANSCRIPTION_CHUNK_LENGTH_S=600` and chunks become ten minutes, no code change. Same code runs on a laptop with a tiny model and on a server with a large one.

## `models.py`

**Problem:** without shared types, every function invents its own format. One returns a dict, another a tuple, a third a list of lists. Nothing fits together.

**Decision:** one small set of data classes that every module speaks. Zero dependencies, no ffmpeg, no whisper, no FastAPI. Just data.

```python
@dataclass(frozen=True, slots=True)
class Segment:
    start: float
    end: float
    text: str
    id: int = 0

    def shifted(self, offset: float) -> "Segment":
        return Segment(self.start + offset, self.end + offset, self.text, self.id)
```

**`shifted` is the most important method in the project.** Three lines. It moves a segment from chunk time to real time. The entire long file feature depends on it.

```python
@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    start: float   # where this window begins in the ORIGINAL file
    end: float
    path: Path
```

**`start` is the field that matters.** It is what a chunk knows about its place in the world, and it is the offset passed to `shifted()`. Without it, reassembly is impossible.

`frozen=True` means these cannot be changed after creation. When stitching, you want certainty that shifting a segment produces a new one rather than secretly mutating something else is holding.

## `asr.py`

**Problem:** if the pipeline imports faster-whisper directly, faster-whisper is welded into the system. Switching vendors means rewriting the pipeline. Testing means downloading a 1.5 GB model.

**Decision:** depend on a promise, not a vendor.

```python
class ASREngine(Protocol):
    def transcribe(self, audio_path: Path, *, language: str | None = None) -> ASRResult: ...
```

That is the whole contract. Anything with a `transcribe` method that takes a path and returns segments counts as an engine. The pipeline never imports faster-whisper. It does not know faster-whisper exists.

**What this buys:**

Adding Groq or Deepgram is one new class and one line in `build_engine`. Nothing else changes.

Testing needs no model at all:

```python
class FakeEngine:
    def transcribe(self, audio_path, *, language=None):
        return ASRResult(segments=[Segment(0.0, 2.0, "fake")], language="en")
```

Five lines replacing 1.5 GB.

**One detail:** the faster-whisper import sits inside `__init__`, not at the top of the file.

```python
def __init__(self, ...):
    from faster_whisper import WhisperModel   # deferred on purpose
```

If it were at the top, just importing this file to get the protocol would load the whole library. Deferring it means the fake engine costs nothing.

**Why faster-whisper:** runs locally with no API key, about four times faster than reference Whisper because it uses a C++ runtime, and works on CPU with int8 quantisation so no GPU is needed.

---

# THE AUDIO FOLDER

Three files because these are three different kinds of failure.

## `audio/probe.py`

**Problem:** you need to know what the file actually is and how long it is. File extensions lie. Anyone can rename a file, and `.ogg` might contain Vorbis or Opus. Extensions also tell you nothing about duration, which chunking needs.

**Decision:** ask the decoder directly.

```python
cmd = [ffprobe, "-print_format", "json", "-show_format", "-show_streams",
       "-select_streams", "a", str(path)]
```

**`-select_streams a` matters more than it looks.** Many mp3 files have embedded album art, which ffprobe reports as a video stream. Without this flag you would read the cover image's properties instead of the audio.

**Duration comes from two possible places:**

```python
duration = _first_float(stream.get("duration"), fmt.get("duration"))
```

Some formats put it on the audio stream, some on the container. Try both. If neither exists, fail, because chunking without a duration is impossible.

**This file fails loudly on purpose:** file missing, file empty, ffprobe not installed, cannot decode, no audio stream, no duration. Each with a clear message. Catching a problem here costs milliseconds. Catching it three steps later costs a minute of wasted processing.

## `audio/normalize.py`

**Problem:** how do you handle mp3 and ogg and flac and m4a and everything else?

**Decision: you do not. You convert everything into one format and forget the rest ever existed.**

```python
cmd = [
    ffmpeg,
    "-nostdin",            # never wait for keyboard input, would hang a server
    "-loglevel", "error",
    "-y",
    "-i", str(src),
    "-vn",                 # drop video AND embedded album art
    "-map", "0:a:0",       # first audio stream only
    "-ac", "1",            # mono
    "-ar", "16000",        # 16 kHz
    "-c:a", "pcm_s16le",   # uncompressed 16 bit PCM
    str(dst),
]
```

**Why these values:**

- **16 kHz** because Whisper is trained on it. Feed 48 kHz and the model resamples internally anyway, so do it once up front.
- **Mono** because speech comes from one mouth. Stereo doubles the data for zero benefit.
- **Uncompressed PCM** because the audio is already decoded, and PCM allows exact sample level seeking, which is what makes chunk cutting precise.

**The payoff:** after this line, nothing in the system asks what format the input was. Search the codebase for "mp3" outside this file and the upload allowlist and you will not find it. `chunk.py`, `asr.py` and `stitch.py` contain zero format awareness.

**The alternative** is a branch per format. That code grows forever, every new format is a new bug, and every downstream function needs format knowledge too.

**The cost** is depending on ffmpeg as a system binary. That is real, and the Dockerfile pins it.

## `audio/chunk.py`

**Problem:** a two hour file cannot go into a model in one piece. Memory grows with length. A crash at minute 90 loses everything. And you get no output at all until it fully finishes.

**Decision:** cut it into five minute windows. But the windows overlap.

### Why overlap

Cut at exactly 300 seconds and you might slice through the middle of a word. That word is now half in each piece, and both halves are unintelligible. The word is destroyed.

```
chunk 0:  |=========================|              0 to 300
chunk 1:                    |=========================|   295 to 595
                            ^^^^^^^^^
                            295 to 300 is in BOTH
```

With overlap, anything damaged at one window's edge sits comfortably in the middle of the next. No word can be lost.

Five seconds is longer than any spoken word plus a pause. Generous on purpose.

### The planning function

```python
def plan_windows(duration, length, overlap):
    if duration <= length:
        return [(0.0, duration)]
    step = length - overlap
    windows = []
    start = 0.0
    while start < duration:
        end = min(start + length, duration)
        windows.append((start, end))
        if end >= duration:
            break
        start += step
    return windows
```

**`step = length - overlap` is the line that creates the overlap.** Windows advance by 295 seconds, not 300. Advance by 300 and there is no overlap at all, which brings the cut word problem straight back.

**Why this function is separated out:** it is pure arithmetic. No files, no ffmpeg. That means it can be tested exhaustively in microseconds. Everything else in this file shells out to ffmpeg and is slow and awkward to test. Separating logic from IO is what makes verification possible.

**Short file shortcut:** if the file fits one window, no cutting happens at all, the normalized wav is used directly. A thirty second clip pays zero chunking cost.

---

# `stitch.py`

**The hardest logic in the project. Twenty five lines. This is the part worth understanding properly.**

## Two problems, created by chunking

**Problem A: every chunk thinks it starts at zero.** The model was handed `chunk_0001.wav` as an isolated file. It has no idea this was carved out of a bigger recording at the 295 second mark. So it reports a word at 5.0 when the truth is 300.

**Problem B: overlap creates duplicates.** That shared five seconds got transcribed twice, once by each chunk. Naive concatenation repeats speech at every single seam.

## Solving A: shift the times

```python
absolute = [[s.shifted(tc.chunk.start) for s in tc.segments] for tc in ordered]
```

Every segment gets its chunk's start added. Chunk 1's word at local 5.0 becomes absolute 300.0. Done.

This is the only reason `Chunk` stores `start`.

## Solving B: seam ownership

**Draw a line at the exact midpoint of every overlap.**

```python
seam = (ordered[i].chunk.end + ordered[i + 1].chunk.start) / 2.0
```

Chunk 0 covers 0 to 300, chunk 1 covers 295 to 595. The shared region is 295 to 300, so the midpoint is 297.5.

**Rule: a segment belongs to whichever chunk's territory its CENTRE falls in.**

```python
center = (seg.start + seg.end) / 2.0
if lo <= center < hi:
    kept.append(seg)
```

### Worked example

A word spoken at 296 to 297 seconds. Both chunks transcribed it.

| | Chunk 0's copy | Chunk 1's copy |
|---|---|---|
| Chunk starts at | 0 | 295 |
| Model reported | 296 to 297 | 1 to 2 |
| After shifting | 296 to 297 | 296 to 297 |
| Centre | 296.5 | 296.5 |
| Territory | up to 297.5 | 297.5 onward |
| Centre inside? | yes | no |
| **Result** | **kept** | **dropped** |

Exactly one copy survives. No gap, no duplicate.

A word later in the overlap, say 299 to 299.5, has a centre of 299.25 which is past 297.5. Now chunk 0's copy is dropped and chunk 1's is kept. Still exactly one, just owned by the other side.

### Why centre and not start

The model does not segment identically when given different audio boundaries. The same word might be reported as 296.8 to 298.1 by one chunk and 297.2 to 298.0 by the other. The centre sits in the middle of actual speech, so it is stable across those differences. An edge is wherever the model happened to cut.

## Why this file has no IO

Look at what it touches: chunk start times, segment times, text strings. Numbers and strings. No files, no ffmpeg, no model.

That is why it gets its own file and why it can be tested completely and instantly.

## Why this is the bug that hides

On a thirty second file there are no seams, because the file fits one window. **A completely broken stitch function passes every manual test you would think to run.** The bug only appears on files longer than five minutes, which is exactly what the assessment asks about.

That is why this got tests and the ffmpeg wrappers did not.

---

# `pipeline.py`

**Problem:** something has to call the steps in order.

**Decision:** one function that orchestrates and contains no logic of its own.

```python
meta = probe(path, settings)
normalized = normalize(path, work / "normalized.wav", settings)
chunks = chunk(normalized, meta, settings, work)

for c in chunks:
    result = engine.transcribe(c.path, language=language)
    transcribed.append(TranscribedChunk(chunk=c, segments=result.segments))

segments = stitch(transcribed)
```

That is the whole system in eight lines.

**Three decisions inside it:**

**Language pinning.** Whisper auto detects language on every call. Without pinning, a quiet chunk in the middle could be detected as a different language and produce garbage. So detect once on the first chunk, then force it for the rest.

```python
if language is None:
    language = result.language
```

**Temp directory with cleanup.** Every run gets its own isolated folder, so two concurrent jobs cannot overwrite each other's `chunk_0001.wav`. The `finally` guarantees cleanup even on a crash. Without it a server slowly fills its disk with abandoned chunks until it dies.

**It receives the engine rather than building one.**

```python
def transcribe_file(path, *, settings, engine: ASREngine, cleanup=True):
```

This is what lets tests pass in a fake engine, and what lets the API load the model once at startup instead of once per job.

---

# `exporters.py`

**Problem:** the transcript needs to come out as text people can use.

**Decision:** one timestamp formatter, four thin renderers.

```python
def _ts(seconds: float, sep: str) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"
```

**Convert to integer milliseconds first.** Working in integers avoids floating point drift that would produce timestamps like `00:00:01,4999999`.

**The `sep` parameter is why this is one function instead of two.** SRT uses a comma before milliseconds, WebVTT uses a dot. Everything else is identical. Get it wrong and subtitle players silently reject the file with no error.

The `EXPORTERS` dict at the bottom maps names to functions, which is why both the CLI and the API can validate `--format srt` with one lookup. Adding a format is one function plus one dict entry.

---

# `cli.py`

**Problem:** the task says "service or script". This is the script half.

**Decision:** keep it thin. Parse arguments, build an engine, call the pipeline, render output. About thirty lines.

**That thinness is the point.** All the real work lives in the pipeline, which is why the API can do the same thing over HTTP while sharing every line of actual logic.

The faster-whisper import is deferred here too, so `transcribe --help` stays instant instead of loading a machine learning library just to print help text.

---

# THE API FOLDER

## The core decision

**Problem:** a ten minute file takes a minute to transcribe. HTTP requests time out. Load balancers cut connections.

**Decision: do not return the transcript. Return a job id and let the client poll.**

```
POST /v1/transcriptions            -> 202, here is your job id
GET  /v1/transcriptions/{id}       -> queued | running | completed | failed
GET  /v1/transcriptions/{id}/result?format=srt
```

**202 rather than 200 is deliberate.** 200 means "here is your result". 202 means "I accepted this, it is not done yet". The status code itself communicates the design.

## `api/schemas.py`

**Problem:** internal data classes and public JSON look similar but are completely different things.

**Decision:** keep them separate.

`models.py` is internal. Rename a field and only your own code cares.

`schemas.py` is the public contract. Every client depends on these exact names. Change one and you break everyone.

If they were the same class, an internal refactor would silently change your public API with no warning.

## `api/jobs.py`

This file answers three of the four design questions.

### Concurrency: the thing people confuse

**Accepting an upload is cheap.** It is disk IO. Ten at once is fine.

**Running a transcription is expensive.** It spawns ffmpeg and occupies the model. Ten at once would collapse a laptop.

**Decision: decouple them.**

```python
self._sem = asyncio.Semaphore(max_concurrency)

async def run(self, job):
    async with self._sem:
        ...
```

Every upload is accepted immediately. The semaphore caps how many actually execute. Ten uploads all succeed, all queue, two process at a time.

**That is the core idea behind every job queue ever built.**

### Keeping the server alive

```python
result = await asyncio.to_thread(transcribe_file, job.source, ...)
```

**Problem:** FastAPI runs on an event loop, a single thread juggling many requests. It works because everything is supposed to be waiting on IO.

Transcription is not IO. It is solid CPU work that never yields. Called directly it would freeze the entire loop, and no other request would be served, not even a health check.

`asyncio.to_thread` moves it to a separate thread so the loop keeps working.

### Retries

**Problem:** some failures are worth retrying and some are not.

**Decision:** split them by whether retrying could possibly help.

```python
except ValueError as exc:
    # bad input, deterministic, retrying gives the same error
    job.status = JobStatus.failed
    return

except Exception as exc:
    # transient, might work next time
    if attempt == self._max_attempts:
        job.status = JobStatus.failed
        return
    await asyncio.sleep(2 ** (attempt - 1))   # 1s, 2s, 4s
```

**Bad input** is empty, corrupt, or has no audio stream. Retrying produces the identical error. Fail immediately with a clear message.

**Transient failure** is ffmpeg being killed, memory spiking, the model glitching. Retry three times with backoff.

**Backoff is exponential rather than fixed** because if the system is already under pressure, retrying immediately makes it worse.

**This is why `probe.py` and `normalize.py` raise `ValueError` specifically.** That choice, made two files earlier, is what makes this policy possible.

### Storage, and an honest limit

```python
class JobStore(Protocol):
    async def create(self, job) -> None: ...
    async def get(self, job_id) -> Job | None: ...
    async def update(self, job) -> None: ...
    async def count_active(self) -> int: ...
```

Same idea as `ASREngine`. Four methods, that is the contract.

**Only the in memory version is built, and that is deliberate.** Jobs die when the process dies. The protocol exists so swapping in Redis or Postgres means writing one class with those four methods and changing nothing else.

Say this limit out loud before anyone asks.

## `api/endpoints.py`

**Routing only, no business logic.**

### Streaming the upload

```python
with dest.open("wb") as out:
    while chunk := await file.read(1024 * 1024):
        written += len(chunk)
        if written > settings.max_upload_bytes:
            raise HTTPException(413, ...)
        out.write(chunk)
```

**Read in one megabyte pieces.** Reading it all at once would put a 500 MB file entirely in RAM. Ten concurrent uploads becomes 5 GB and the server dies. Streaming keeps memory flat regardless of file size.

**Check the size limit during the transfer, not after.** Afterwards means you already paid to receive the whole thing. During means you reject a 5 GB upload after 200 MB.

### Two layers of validation

The extension allowlist is a cheap first filter that rejects an obvious `.exe` before writing anything. But **ffprobe is the real validator**, because extensions lie.

## `api/main.py`

**Problem:** loading a Whisper model takes seconds and hundreds of megabytes.

**Decision:** load it once at startup, share it forever.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = build_engine(settings)     # once
    app.state.engine = engine
```

Per request loading would make every transcription pay that cost, and ten concurrent requests would try to hold ten copies in memory.

The `/v1` prefix is versioning. When a breaking change is eventually needed, add `/v2` and existing clients keep working. Free now, impossible to add gracefully later.

---

# THE TESTS

**The principle: you cannot test everything, so test what can be silently wrong.**

**ffmpeg wrappers get no unit tests.** If ffmpeg fails it fails loudly with an exit code and an error message. There is no subtle wrong answer.

**Stitching gets exhaustive tests.** It can produce output that looks completely fine but is wrong. Timestamps drifted by minutes. Words missing at every seam. Words duplicated at every seam. None of that raises an error, and none of it shows on a short file.

That asymmetry is the whole basis for what got tested.

## The tests that matter most

**`test_every_point_is_covered`** walks the whole timeline second by second and asserts no instant falls in a gap between windows. If someone changes the step arithmetic, this fails instantly.

**`test_overlapping_word_kept_exactly_once`** proves dedup works.

**`test_word_past_the_seam_goes_to_the_right_chunk`** is the mirror, and the more important one. It is easy to write logic that removes duplicates but accidentally drops the word entirely when its centre falls on the far side. This proves ownership transfers rather than the word vanishing.

## The failure that taught something

One test failed on the first run. The fixture put each word at chunk local time 1.0, which meant the word in chunk 1 landed at absolute 296.5, before chunk 1's own seam at 297.5. Stitch correctly assigned that region to chunk 0 and dropped chunk 1's copy.

**The logic was right. The test data was unrealistic.** In real audio, chunk 0 would also have transcribed that word, and chunk 0's copy is the survivor. The fixture simulated an impossible scenario.

Fixed by moving the words clear of the overlap.




---

# QUICK REFERENCE

```powershell
pip install -e ".[whisper,api,dev]"

transcribe audio.mp3 --format srt
pytest -v
uvicorn transcription.api.main:app --reload
python tests/test_load.py --file audio.ogg --users 5
```

| Question they ask | Answer lives in |
|---|---|
| Accepts an audio file | `cli.py` and `api/endpoints.py` |
| Transcribes to text | `asr.py` |
| Timestamps per segment | `models.py` and `exporters.py` |
| Different formats | `audio/normalize.py` |
| Long files | `audio/chunk.py` and `stitch.py` |
| Concurrent uploads | `api/jobs.py`, the semaphore |
| Storage | `api/jobs.py`, the JobStore protocol |
| Retries | `api/jobs.py`, the two except blocks |
| Expose as API | `api/endpoints.py` |
