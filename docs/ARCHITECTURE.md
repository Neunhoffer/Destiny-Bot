# Destiny Bot: architecture and module contracts

Destiny Bot is a private, offline-first study reader. It opens textbooks and
class materials (PDF, EPUB, Word, PowerPoint, HTML, text, photos of handouts),
cleans them up for listening, and reads them aloud with a natural voice while
highlighting the sentence being read. Optional Claude-powered helpers explain,
summarize and quiz.

## Principles

1. **Private by default.** Nothing leaves the computer unless the user turns on an
   online feature in Settings. The local web server binds to `127.0.0.1` only. The UI
   loads no external scripts, fonts, or images. There is no telemetry or analytics.
2. **Online features are explicit and labelled.** These are Edge online voices
   (text goes to Microsoft) and Claude AI helpers (text goes to Anthropic). Both are
   gated by switches in `Settings` that default to off. The UI marks them with a
   cloud badge.
3. **One private data folder.** Everything is stored under `config.data_dir()`, which is
   `%LOCALAPPDATA%\Athena` on Windows, or `DESTINY_HOME` if set. On POSIX the folder is created
   with owner-only (0700) permissions.
   "Erase all my data", or `destiny wipe`, removes all traces.
4. **Faithful text.** Display text stays faithful to the book. Speech-only changes
   live in the speech layer: expanding "e.g.", dropping citation brackets, and so on.
   AI never silently rewrites the book.
5. **Never crash on a bad file.** Loaders raise `IngestError` with a friendly
   message. Partial problems become `Document.warnings`.

## Layout

```
src/destiny_bot/
  __init__.py          version
  models.py            Document / Section / Block / ReadingOptions / Utterance  (shared contract)
  config.py            data folder paths + Settings load/save                    (shared contract)
  ingest/
    __init__.py        load_document() dispatcher, IngestError, SUPPORTED_EXTENSIONS
    common.py          clean_display_text, sections_from_blocks, finalize_sections, doc ids
    pdf.py             load_pdf()
    epub.py            load_epub()
    html.py            load_html() + html_to_blocks() (shared by epub)
    docx.py            load_docx()
    pptx.py            load_pptx()
    text.py            load_text_file(), document_from_text()
    image.py           load_image()   (OCR)
    ocr.py             ocr_available(), ocr_image_bytes()
  speech/
    __init__.py        re-exports normalize_for_speech, split_sentences, plan_section, section_plain_text
    normalize.py       text -> speakable text
    segment.py         abbreviation-aware sentence splitting + long-sentence chunking
    plan.py            Section -> list[Utterance]
  tts/
    base.py            TTSEngine ABC, Voice, AudioClip, TTSError, EngineUnavailable (shared contract)
    __init__.py        engine registry: get_engine(), all_engines(), resolve_engine()
    kokoro_engine.py   the most natural offline voices (kokoro-onnx, 24 kHz)
    piper_engine.py    fast natural offline voices (piper-tts)
    system_engine.py   OS voices: macOS `say`, Linux espeak-ng, Windows SAPI 5 (comtypes, PowerShell fallback)
    edge_engine.py     Microsoft Edge online voices (edge-tts), online=True
    browser_engine.py  placeholder: the browser's Web Speech API speaks in the UI
    catalog.py         curated downloadable Piper voices + downloader
    cache.py           on-disk LRU audio cache
  audio.py             WAV/MP3 helpers: concat, silence, mp3 encode, play on speakers (CLI)
  export.py            export sections to MP3/WAV audiobook files
  ai/
    __init__.py        re-exports Assistant, AIUnavailable, AIError, QuizQuestion
    prompts.py         system prompts
    claude.py          Assistant (Claude via the official anthropic SDK)
  library.py           local library: imported books, progress, bookmarks
  jobs.py              background jobs (import, export, voice download) with progress
  server.py            FastAPI app (local web UI + JSON API, see docs/API.md)
  cli.py               `destiny` command
  __main__.py          `python -m destiny_bot`
  web/                 index.html, app.js, styles.css, icon.svg   (no build step, no CDN)
tests/                 pytest suite (fixtures are generated in code; no binary fixtures committed)
scripts/               start-windows.bat, start-mac.command, start-linux.sh
```

## Module contracts

The signatures below are the contract between modules. Implementations may add
private helpers and extra keyword-only parameters with defaults, but must not
change these.

### ingest

```python
# ingest/__init__.py (written)
def load_document(path: Path, *, source_name: str | None = None, ocr: bool = True,
                  progress: ProgressFn | None = None) -> Document
def document_from_text(text: str, title: str, *, source_type: str = "paste") -> Document

# each loader
def load_pdf(path: Path, *, source_name: str, ocr: bool = True, progress: ProgressFn | None = None) -> Document
def load_epub(path: Path, *, source_name: str, progress: ProgressFn | None = None) -> Document
def load_html(path: Path, *, source_name: str, progress: ProgressFn | None = None) -> Document
def load_docx(path: Path, *, source_name: str, progress: ProgressFn | None = None) -> Document
def load_pptx(path: Path, *, source_name: str, progress: ProgressFn | None = None) -> Document
def load_text_file(path: Path, *, source_name: str, progress: ProgressFn | None = None) -> Document
def document_from_text(text: str, title: str, *, source_type: str = "paste") -> Document   # in text.py
def load_image(path: Path, *, source_name: str, progress: ProgressFn | None = None) -> Document

# html.py: reused by epub.py
def html_to_blocks(html: str | bytes, *, page: int | None = None) -> tuple[list[Block], str | None]  # (blocks, <title>)

# ocr.py
def ocr_available() -> tuple[bool, str]             # (usable, reason / how to install Tesseract)
def ocr_image_bytes(data: bytes, *, lang: str = "eng") -> str   # raises IngestError if unavailable
```

Rules for every loader:
- `Document.id` = `common.doc_id_for_file(path)`, the hash of the source bytes.
  The same file always gets the same id.
- Every `Block.text` goes through `common.clean_display_text`, except CODE blocks,
  which keep their line breaks but get `fix_characters`.
- Headings are kept as `HEADING` blocks with `level`. Sections are built with
  `common.sections_from_blocks` or from the file's own table of contents. Finish
  with `common.finalize_sections`, which assigns ids `s0..sN`.
- The section's own heading is the first block of that section, when one exists.
- `Block.page` is set for PDF (page number) and PPTX (slide number).
- `Library.import_file` reports `(1.0, "Saving to your library")` to the progress callback just
  before it stores the book. That callback is the import job's last cancel checkpoint, so an
  "Erase all my data" that happens during a slow import can't bring the book back.
- Recoverable problems are added to `Document.warnings`. Examples: "Pages 3, 9 had
  no text layer and were read with OCR" and "Pages 12-14 are scanned images; install
  Tesseract to read them".

### speech

```python
def normalize_for_speech(text: str, options: ReadingOptions | None = None) -> str
def split_sentences(text: str) -> list[tuple[int, int]]   # (start, end) spans covering the sentences, trimmed
def plan_section(section: Section, options: ReadingOptions | None = None) -> list[Utterance]
def section_plain_text(section: Section, *, max_chars: int | None = None) -> str   # readable text for AI context
```

`plan_section` guarantees the following:
- Utterance `index` values are 0..n-1 in reading order.
- `speech` is never empty or whitespace-only.
- Each utterance is at most `options.max_chars` characters. Longer sentences split
  at `; : ,` or spaces.
- For text blocks, `blocks[block].text[start:end]` is exactly the displayed span
  being spoken.
- Every utterance has a non-zero `pause_after_ms`. Voice output is trimmed of
  leading and trailing silence, so this pause is the only gap between utterances.
  It is 220 ms between sentences inside a block, 60 ms after a chunk of a split
  sentence, 450 ms after a paragraph, quote, caption, footnote, figure, equation
  or code block, 300 ms after a list item or table row, and 750 ms after a
  heading or section-title announcement.
- Blocks that options say to skip produce no utterances.
- If the section's first block is not a HEADING and `announce_section_title` is set,
  utterance 0 is `block=-1, start=0, end=0` and speaks the title.
- Tables in `rows` mode give one utterance announcing "Table with N rows and M
  columns", then one utterance per row: "Row 1: Name, Alice; Age, 30" using header
  cells as labels when the first row looks like a header. These highlight the whole
  table block (start=0, end=len(text)).

### tts

```python
# tts/__init__.py
ENGINE_NAMES = ("piper", "system", "edge", "browser")
def get_engine(name: str) -> TTSEngine            # singleton per name; KeyError if unknown
def all_engines() -> list[TTSEngine]
def resolve_engine(preferred: str, *, online_allowed: bool) -> TTSEngine
    # "auto" -> first available of piper (with an installed voice), system, browser.
    # An online engine is never returned unless online_allowed is True.

# tts/catalog.py
class CatalogVoice(BaseModel): id, name, language, quality, size_mb, description
CATALOG: list[CatalogVoice]                       # curated English voices; en_US-lessac-medium recommended
def installed_piper_voices(voices_dir: Path | None = None) -> list[Path]   # *.onnx with a matching .onnx.json
def download_piper_voice(voice_id: str, *, voices_dir: Path | None = None,
                         progress: Callable[[float, str], None] | None = None,
                         base_url: str | None = None, cancel: threading.Event | None = None) -> Path

# tts/cache.py
class AudioCache:
    def __init__(self, root: Path, max_bytes: int = 500 * 1024 * 1024)
    def get_or_create(self, key_parts: tuple[str, ...], make: Callable[[], AudioClip]) -> AudioClip
    def clear(self) -> None

# audio.py
def wav_bytes(pcm: bytes, sample_rate: int, channels: int = 1, sampwidth: int = 2) -> bytes
def read_wav(data: bytes) -> tuple[bytes, int, int, int]      # pcm, sample_rate, channels, sampwidth
def silence_pcm(ms: int, sample_rate: int, channels: int = 1, sampwidth: int = 2) -> bytes
def concat_clips(clips: list[tuple[AudioClip, int]]) -> AudioClip   # (clip, pause_after_ms); wav+wav -> wav, mp3 -> mp3
def encode_mp3(wav: bytes, *, bitrate_kbps: int = 64) -> bytes  # lameenc, mono
def trim_silence(clip: AudioClip, *, threshold_db: float = -45.0, keep_ms: int = 25) -> AudioClip  # WAV only
def play_clip(clip: AudioClip) -> None                         # blocking playback for the CLI

# export.py
def export_audiobook(doc: Document, *, engine: TTSEngine, voice: str | None,
                     options: ReadingOptions, out_dir: Path, section_ids: list[str] | None = None,
                     fmt: Literal["mp3", "wav"] = "mp3", rate: float = 1.0,
                     progress: Callable[[float, str], None] | None = None,
                     cancel: threading.Event | None = None) -> list[Path]
    # one file per section: "01 - Chapter 1 Cells.mp3" (safe file names), plus "playlist.m3u"
```

### ai

```python
class AIUnavailable(RuntimeError)   # AI turned off / no API key / anthropic not installed (friendly message)
class AIError(RuntimeError)         # the API call failed (friendly message)
class QuizQuestion(BaseModel): question: str; choices: list[str]; answer_index: int; explanation: str

class Assistant:
    def __init__(self, settings: Settings, *, client: Any | None = None)   # client injectable for tests
    def explain(self, passage: str, *, context: str, book_title: str) -> str
    def ask(self, question: str, *, context: str, book_title: str,
            history: list[dict[str, str]] | None = None) -> str
    def summarize(self, section_text: str, *, section_title: str, book_title: str,
                  length: Literal["short", "detailed"] = "short") -> str
    def quiz(self, section_text: str, *, section_title: str, count: int = 5) -> list[QuizQuestion]
    def narrate_block(self, block: Block, *, context: str = "") -> str   # tables/equations -> natural speech
    def describe_image(self, png: bytes, *, context: str = "") -> str   # e.g. a rendered PDF page
```

All answers are plain text meant to be read aloud, with no Markdown. Book text is
untrusted data and is wrapped in tags. The system prompt tells Claude to treat it
only as material to study.

### library

```python
class Progress(BaseModel): section_id: str; utterance: int = 0; updated_at: str
class Bookmark(BaseModel): id: str; section_id: str; utterance: int; label: str; created_at: str
class BookSummary(BaseModel): id; title; author; source_name; source_type; added_at; section_count;
                             page_count; progress: Progress | None
class Library:
    def __init__(self, root: Path | None = None)        # default config.library_dir()
    def import_file(self, path: Path, *, original_name: str | None = None, progress=None) -> Document
    def import_text(self, title: str, text: str) -> Document
    def list(self) -> list[BookSummary]                 # most recently opened/added first
    def get(self, doc_id: str) -> Document              # KeyError if missing
    def summary(self, doc_id: str) -> BookSummary
    def source_file(self, doc_id: str) -> Path | None   # the stored copy of the original file
    def delete(self, doc_id: str) -> None
    def get_progress(self, doc_id: str) -> Progress | None
    def set_progress(self, doc_id: str, section_id: str, utterance: int) -> Progress
    def list_bookmarks(self, doc_id: str) -> list[Bookmark]
    def add_bookmark(self, doc_id: str, section_id: str, utterance: int, label: str) -> Bookmark
    def delete_bookmark(self, doc_id: str, bookmark_id: str) -> None
```

On-disk layout: `library/<doc_id>/document.json`, `library/<doc_id>/meta.json`
(added_at, last_opened, progress, bookmarks), and `library/<doc_id>/source<ext>`.
All writes are atomic (write a temp file, then `os.replace`).
