# N-Pass contract (2026-11-22)

What the pieces promise each other. Read this and `npass/types.py` before writing code.
The lead owns this file, `npass/types.py`, `npass/settings.py`, `npass/app.py`, `pyproject.toml`.
Stations do not edit those; ask for a change in your final report.

## What the program does
Mic (48 kHz) -> look-ahead buffer (350-600 ms, default 550 since the 2026-11-23 measurements) -> cutter -> virtual cable -> OBS.
A recogniser listens to the buffered audio and orders cuts over blocked words before they play.
Optional: Spotify position + synced lyrics mark "danger windows" (lines containing a blocked word).

Hard rules, for every station:
- Recognition is local. No audio, and no text of what was said, ever leaves the PC. The only
  network calls in the whole program: Spotify Web API (currently playing, token refresh) and
  LRCLIB lyrics. Nothing else, no telemetry, no update check. (The one-time setup command
  `python -m npass.detect download` also fetches the speech models from huggingface.co and
  alphacephei.com; the running app never does.)
- The output is the user's own voice with stretches silenced or dipped. Never synthesise, clone or
  insert any voice or beep. No TTS dependency anywhere.
- Never print, log or put in an exception message: Spotify tokens, the auth code, the PKCE verifier.
- The blocked word list is the user's (`settings.load_words()`). It ships empty. Never hard-code a
  slur in code, tests, comments or docs: tests use the stand-in word `pineapple` (and `banana`).
- Plain English in anything the user reads: they are not a programmer. No em dashes in user-facing
  text. Dates absolute (2026-11-22).
- Tests set `NPASS_DATA_DIR` to a temp folder (pytest `tmp_path`); never touch `%LOCALAPPDATA%\NPass`.
  Tests never open real audio devices, never call the network, never need the GPU unless marked
  `@pytest.mark.gpu`.
- Python 3.12, the project venv: run things with `uv run python ...` / `uv run pytest ...` from the
  project root. Need a package that is not in `pyproject.toml`? `uv pip install <pkg>` into the venv
  and name it in your report (do NOT edit pyproject.toml or run `uv add`: several stations share it).
- Several stations work in this one folder at once. Edit only the paths your brief gives you.
  Do not run `git add`/`git commit`: the lead commits.
- Write multi-line scripts with your Write tool and run the file; no bash heredocs.

## Time base
One clock: the sample index at 48 kHz since the engine started (`int`). Sample 0 is the first mic
sample captured. "Playhead" = the index of the sample being handed to the cable right now
= capture index - look-ahead. Lyrics convert Spotify's position onto this clock.

## Pieces and their public faces

### `npass/audio/` (station: audio)
```python
class LookaheadProcessor:                      # pure, no devices: used live AND by the offline harness
    def __init__(self, lookahead_ms, fade_ms=15, mode="silence", dip_db=-30.0): ...
    def process(self, block: np.ndarray) -> np.ndarray   # mono float32 in, same length out, delayed by exactly the look-ahead
    def submit_cut(self, order: CutOrder) -> float       # thread-safe; returns late_ms (0 if on time). Same cut_id = replace/extend
    def set_lookahead(self, ms) -> None                  # glitch-free (fade), takes effect within 1 s
    def set_mode(self, mode, dip_db, fade_ms) -> None
    @property
    def captured(self) -> int                            # samples taken in so far
    @property
    def playhead(self) -> int                            # captured - look-ahead (may be negative at start)
    @property
    def cutting(self) -> bool

class AudioEngine:
    def __init__(self, settings: Settings, tap: AudioTap | None = None): ...
    def start(self) -> None        # never raises for a missing device: reports it in status().warnings and retries
    def stop(self) -> None
    def submit_cut(self, order: CutOrder) -> float
    def set_lookahead(self, ms) -> None
    def set_mode(self, mode, dip_db, fade_ms) -> None
    def status(self) -> EngineStatus
    processor: LookaheadProcessor

def list_inputs() -> list[str]; def list_outputs() -> list[str]
def find_cable() -> str | None    # name of the virtual cable's playback end, None when not installed
```
Cut shape: gain goes 1 -> 0 over `fade_ms` ENDING at `order.start`, stays 0 (or `dip_db`) to
`order.end`, then 0 -> 1 over `fade_ms`. Raised-cosine fades. Overlapping cuts merge. A cut whose
start already played is applied from "now" with a fade and its lateness returned.

### `npass/detect/` + `npass/match.py` (station: detect)
```python
class Detector:
    def __init__(self, settings, words: Callable[[], list[str]], windows: WindowsProvider,
                 submit_cut: SubmitCut, on_catch: Callable[[CutOrder], None],
                 recogniser: Recogniser | None = None): ...   # None = pick by settings.engine
    def start(self) -> None; def stop(self) -> None
    def feed(self, block48k: np.ndarray, start_sample: int) -> None   # the AudioTap; only queues
    def verified_upto(self) -> int      # every sample before this index has been checked
    def health(self) -> Health
    def engine_name(self) -> str        # e.g. "GPU (faster-whisper small.en)" / "CPU (Vosk)"
    def last_pass_ms(self) -> float     # how long the last recognition pass took

class SafetyGuard:                      # pure; the app calls tick() every audio block from its own thread
    def __init__(self, submit_cut: SubmitCut, margin_ms: float = 60, ahead_ms: float = 0,
                 bridge_ms: float = 0): ...   # the app uses ahead_ms=250, bridge_ms=300
    # tick() also takes captured=<capture index> so a mute never reaches past recorded audio
    def tick(self, playhead: int, verified_upto: int, health: Health,
             windows: list[DangerWindow]) -> Health
```
Guard rule: if the recogniser has not verified audio up to `playhead + margin` (or health is
DOWN/STARTING), then for every danger window overlapping the unverified, about-to-play span, submit a
FAILSAFE CutOrder covering that overlap (extend it block by block with the same cut_id). Outside
danger windows: no cut, return LATE/DOWN so the window shows a warning.
Recogniser backends: `WhisperRecogniser` (faster-whisper, GPU) and `VoskRecogniser` (CPU), both
implementing `types.Recogniser`. A scripted `FakeRecogniser` for tests.

### `npass/lyrics/` (station: lyrics)
```python
class LyricService:
    def __init__(self, settings, words: Callable[[], list[str]], now_sample: Callable[[], int]): ...
    def start(self) -> None; def stop(self) -> None
    def windows(self, from_sample: int, to_sample: int) -> list[DangerWindow]   # the WindowsProvider
    def track(self) -> TrackInfo | None       # position interpolated to now
    def status(self) -> str                   # one plain-English line for the window
    def logged_in(self) -> bool
    def login(self) -> None                   # opens the browser (PKCE), blocks until done or 120 s
    def logout(self) -> None                  # deletes the token file
```
`now_sample()` is the CAPTURE index (what the user is singing right now lines up with what Spotify
is playing right now, since they hear the music live).

### `npass/ui/`, `npass/catchlog.py`, `npass/obs_sync.py` (station: app)
The window talks only to this object (the lead implements the real one in `npass/app.py`; the app
station writes a `FakeCore` for its tests and a demo mode):
```python
class Core(Protocol):
    def snapshot(self) -> dict     # keys below; cheap, called ~15x/s
    def set_enabled(self, on: bool) -> None
    def set_lookahead(self, ms: int) -> None
    def set_removal(self, mode: str) -> None            # "silence" | "dip"
    def set_engine(self, engine: str) -> None           # "auto" | "gpu" | "cpu"
    def inputs(self) -> list[str]; def set_input(self, name: str) -> None
    def words(self) -> list[str]; def set_words(self, words: list[str]) -> None
    def set_lyrics_enabled(self, on: bool) -> None
    def spotify_login(self) -> None; def spotify_logout(self) -> None
    def set_spotify_client_id(self, cid: str) -> None
    def recent_catches(self) -> list[Catch]
    def quit(self) -> None
```
`snapshot()` keys: `enabled` bool, `engine` EngineStatus, `health` Health, `engine_name` str,
`pass_ms` float, `caught` int (this session), `lyrics_status` str, `track` TrackInfo|None,
`in_window` bool, `cable_found` bool, `warnings` list[str], `lookahead_ms` int, `removal_mode` str,
`lyrics_enabled` bool, `spotify_logged_in` bool, `spotify_client_id` str, `hotkey` str.

`catchlog.log_catch(catch: Catch) -> None` appends one line to `settings.catch_log_path()`:
```
2026-11-22 21:14:03 | caught "<word>" (0.71, lyric window) | Artist - Song @ 1:42
```
(no song known: the last part is `no song`; a late cut adds ` | late 35 ms`).

## How the lead wires it (`npass/app.py`)
engine = AudioEngine(settings, tap=detector.feed); detector gets `lyrics.windows`,
`engine.submit_cut`; a 10 ms guard thread calls `guard.tick(processor.playhead,
detector.verified_upto(), detector.health(), lyrics.windows(playhead, captured))`.
With `enabled` off: audio still passes through the same buffer (delay stays constant so OBS sync
never jumps), no cuts are made.
