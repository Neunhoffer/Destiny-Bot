"""The ``athena`` command.

    athena                        open the reader in your browser (same as ``athena serve``)
    athena serve [--port N] [--no-browser] [--host H]
    athena app [--port N]         open the reader in its own window
    athena shortcut [--no-start-menu]
    athena self-test [--quick]
    athena import FILE...
    athena list
    athena toc BOOK
    athena text BOOK|FILE [--section ID] [--speech]
    athena read BOOK|FILE [--section ID] [--engine E] [--voice V] [--rate R]
    athena export BOOK [--section ID ...] [--format mp3|wav] [--out DIR] [--engine E] [--voice V] [--rate R]
    athena voices [--engine E] | athena voices catalog | athena voices download VOICE_ID
    athena ai-key [--remove]
    athena doctor
    athena wipe [--yes]

BOOK is a library id, the first few (4+) characters of one, or part of the title.
A FILE that exists is imported into the library first.
"""

from __future__ import annotations

import argparse
import getpass
import importlib
import os
import platform
import re
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

from . import __version__, config
from .ingest import IngestError
from .jobs import is_friendly_error
from .library import Library, is_valid_doc_id, wipe_user_data
from .models import Block, BlockKind, Document, Section, Utterance
from .tts.base import AudioClip, TTSEngine

ONLINE_TTS_OFF = "Online voices are turned off in Settings (they send text to Microsoft)."
DEFAULT_VOICE_ID = "en_US-lessac-medium"
_HEX_RE = re.compile(r"[0-9a-f]+")
_UNSAFE_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')


class CLIError(Exception):
    """An expected problem; its message is printed without a traceback."""


# -- lazily imported modules (heavy or optional). Tests replace these accessors with fakes.


def _tts() -> Any:
    from . import tts

    return tts


def _speech() -> Any:
    from . import speech

    return speech


def _audio() -> Any:
    from . import audio

    return audio


def _audio_cache_class() -> Any:
    from .tts.cache import AudioCache

    return AudioCache


def _export_audiobook() -> Callable[..., list[Path]]:
    from .export import export_audiobook

    return export_audiobook


def _catalog() -> Any:
    from .tts import catalog

    return catalog


def _ocr_status() -> tuple[bool, str]:
    try:
        from .ingest.ocr import ocr_available
    except ImportError as exc:
        return False, f"not installed ({exc})"
    try:
        ok, reason = ocr_available()
    except Exception as exc:
        return False, f"couldn't check: {exc}"
    return bool(ok), str(reason or "")


# -- output helpers


def _make_output_safe() -> None:
    """Never crash on a book title the console can't display (e.g. Windows code pages)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def _isatty(stream: TextIO) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


def _err(message: str) -> None:
    print(message, file=sys.stderr)


class _ProgressBar:
    """A one-line progress bar on stderr (silent when stderr isn't a terminal)."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stderr
        self.enabled = _isatty(self.stream)
        self._last = ""

    def __call__(self, fraction: float, message: str = "") -> None:
        if not self.enabled:
            return
        width = shutil.get_terminal_size((80, 20)).columns
        pct = int(max(0.0, min(1.0, float(fraction))) * 100)
        filled = pct // 5
        line = f"[{'#' * filled}{'-' * (20 - filled)}] {pct:3d}%  {message}"[: max(10, width - 1)]
        self.stream.write("\r" + line.ljust(len(self._last)))
        self.stream.flush()
        self._last = line

    def finish(self) -> None:
        if self.enabled and self._last:
            self.stream.write("\r" + " " * len(self._last) + "\r")
            self.stream.flush()
            self._last = ""


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _book_line(doc_title: str, sections: int, pages: int | None) -> str:
    parts = [_plural(sections, "section")]
    if pages:
        parts.append(_plural(pages, "page"))
    return f"{doc_title} ({', '.join(parts)})"


def _display_block(block: Block) -> str:
    text = block.text
    if block.kind == BlockKind.TABLE and block.rows:
        text = "\n".join(" | ".join(cell.strip() for cell in row) for row in block.rows)
    elif block.kind == BlockKind.HEADING:
        text = "#" * max(1, min(6, block.level or 1)) + " " + text
    elif block.kind == BlockKind.LIST_ITEM:
        text = "- " + text
    elif block.kind == BlockKind.QUOTE:
        text = "> " + text
    elif block.kind == BlockKind.FIGURE:
        text = f"[Figure: {text}]" if text else "[Figure]"
    elif block.kind == BlockKind.FOOTNOTE:
        text = f"[Note: {text}]"
    if block.meta.get("speaker_notes"):
        text = "[Speaker notes] " + text
    return text


def _utterance_text(section: Section, utterance: Utterance) -> str:
    """What's on screen for an utterance: the display span when there is one, else the spoken text."""
    if 0 <= utterance.block < len(section.blocks):
        block = section.blocks[utterance.block]
        if block.kind != BlockKind.TABLE:
            span = block.text[utterance.start : utterance.end].strip()
            if span:
                return span
    return utterance.speech


def _safe_folder_name(title: str) -> str:
    from .export import safe_filename  # also renames Windows device names such as "Con" or "Aux.2"

    if not _UNSAFE_NAME_CHARS.sub("", title).strip(" ."):
        return "Athena export"
    return safe_filename(_UNSAFE_NAME_CHARS.sub("_", title))


# -- books


def _import(lib: Library, path: Path, *, quiet: bool = False) -> Document:
    bar = _ProgressBar()
    try:
        doc = lib.import_file(path, progress=bar)
    finally:
        bar.finish()
    if not quiet:
        _print_import(doc)
    return doc


def _print_import(doc: Document) -> None:
    print(f"Imported: {_book_line(doc.title, len(doc.sections), doc.page_count)}  [id {doc.id}]")
    for warning in doc.warnings:
        print(f"  Note: {warning}")


def resolve_book(lib: Library, ref: str) -> Document:
    """A book from an id, a unique id prefix (4+ chars), part of its title, or a file to import."""
    path = Path(ref).expanduser()
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    if is_file:
        doc = _import(lib, path, quiet=True)
        print(f"Using: {_book_line(doc.title, len(doc.sections), doc.page_count)}  [id {doc.id}]")
        return doc

    needle = ref.strip().lower()
    if not needle:
        raise CLIError("Say which book: give its id or part of its title (see 'athena list').")
    if is_valid_doc_id(needle):
        try:
            return lib.get(needle)
        except KeyError:
            pass
    books = lib.list()
    if len(needle) >= 4 and _HEX_RE.fullmatch(needle):
        by_id = [b for b in books if b.id.startswith(needle)]
        if len(by_id) == 1:
            return _get(lib, by_id[0].id)
        if len(by_id) > 1:
            raise CLIError(_ambiguous(ref, by_id))
    by_title = [b for b in books if needle in b.title.lower()]
    exact = [b for b in by_title if b.title.lower() == needle]
    if len(by_title) == 1 or len(exact) == 1:
        return _get(lib, (exact or by_title)[0].id)
    if by_title:
        raise CLIError(_ambiguous(ref, by_title))
    if path.suffix:
        raise CLIError(f"There's no file '{ref}' and no book in your library matches it.")
    raise CLIError(f"No book in your library matches '{ref}'. See your books with: athena list")


def _get(lib: Library, doc_id: str) -> Document:
    try:
        return lib.get(doc_id)
    except KeyError:
        raise CLIError("That book's files are damaged. Delete it and import it again.") from None


def _ambiguous(ref: str, matches: Sequence[Any]) -> str:
    lines = [f"'{ref}' matches more than one book. Use more of the title or the id:"]
    lines += [f"  {b.id}  {b.title}" for b in matches[:20]]
    if len(matches) > 20:
        lines.append(f"  ... and {len(matches) - 20} more")
    return "\n".join(lines)


def _section_index(doc: Document, section_id: str) -> int:
    for i, section in enumerate(doc.sections):
        if section.id == section_id:
            return i
    raise CLIError(f"'{doc.title}' has no section '{section_id}'. See the sections with: athena toc {doc.id}")


# -- voices


def _choose_engine(name: str | None, settings: config.Settings) -> TTSEngine:
    tts = _tts()
    explicit = bool(name) and name != "auto"
    if explicit:
        try:
            engine = tts.get_engine(name)
        except KeyError:
            names = ", ".join(getattr(tts, "ENGINE_NAMES", ()))
            hint = f" Choose from: {names}." if names else ""
            raise CLIError(f"There's no voice engine called '{name}'.{hint}") from None
    else:
        engine = tts.resolve_engine(settings.tts_engine, online_allowed=settings.online_tts_allowed)
    if engine.name == "browser":
        recommended = getattr(_safe_catalog(), "RECOMMENDED_VOICE", DEFAULT_VOICE_ID)
        prefix = (
            "The browser's voice only works in the web reader."
            if explicit
            else "No voice that works outside the browser is installed."
        )
        raise CLIError(f"{prefix} Download a natural offline voice with:\n  athena voices download {recommended}")
    if engine.online and not settings.online_tts_allowed:
        raise CLIError(ONLINE_TTS_OFF + " You can turn them on in the reader's Settings.")
    try:
        available, reason = engine.is_available()
    except Exception as exc:
        available, reason = False, str(exc)
    if not available:
        raise CLIError(reason or f"The {engine.label} voice isn't available right now.")
    return engine


def _pick_voice(engine: TTSEngine, requested: str | None, settings: config.Settings) -> str:
    """The requested voice (which must be installed), else the saved one if installed, else the default."""
    try:
        installed = [v.id for v in engine.list_voices() if v.installed]
    except Exception:
        installed = []
    if requested:
        if requested in installed:
            return requested
        listing = f" Installed voices: {', '.join(installed[:12])}." if installed else ""
        raise CLIError(f"The voice '{requested}' isn't installed for {engine.label}.{listing} See: athena voices")
    saved = settings.voices.get(engine.name)
    if saved and saved in installed:
        return saved
    try:
        default = engine.default_voice()
    except Exception:
        default = None
    if not default:
        recommended = getattr(_safe_catalog(), "RECOMMENDED_VOICE", DEFAULT_VOICE_ID)
        raise CLIError(
            f"{engine.label} has no voices installed. Download one with:\n  athena voices download {recommended}"
        )
    return default


def _safe_catalog() -> Any:
    try:
        return _catalog()
    except ImportError:
        return None


def _clamp_rate(rate: float) -> float:
    try:
        clamp = _tts().clamp_rate
    except (ImportError, AttributeError):
        return rate
    return float(clamp(rate))


def _rate_key(rate: float) -> str:
    return f"{rate:.2f}"


# -- commands


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        from .server import DEFAULT_HOST, run
    except ImportError as exc:
        raise CLIError(f"The web reader can't start because a part is missing ({exc}). Reinstall Athena.")
    host = args.host or DEFAULT_HOST
    try:
        run(host=host, port=args.port, open_browser=not args.no_browser, allow_remote=args.host is not None)
    except ValueError as exc:
        raise CLIError(str(exc)) from None
    return 0


def _desktop() -> Any:
    try:
        # import_module, not "from . import desktop": that returns the package attribute
        # even when sys.modules holds another module (the test fakes).
        desktop = importlib.import_module(f"{__package__}.desktop")
    except ImportError as exc:
        raise CLIError(
            f"The desktop app isn't available in this copy of Athena ({exc}). "
            "Run 'athena' to open the reader in your browser instead."
        ) from None
    return desktop


def _cmd_app(args: argparse.Namespace) -> int:
    desktop = _desktop()
    try:
        run_app = desktop.run_app
    except AttributeError:
        raise CLIError("The desktop app isn't ready in this copy of Athena. Run 'athena' instead.") from None
    return int(run_app(port=args.port) or 0)


def _cmd_shortcut(args: argparse.Namespace) -> int:
    desktop = _desktop()
    try:
        create_shortcuts = desktop.create_shortcuts
    except AttributeError:
        raise CLIError("Creating shortcuts isn't available in this copy of Athena.") from None
    paths = list(create_shortcuts(desktop=True, start_menu=not args.no_start_menu) or [])
    if not paths:
        print("No shortcuts were created.")
    for path in paths:
        print(f"Created {path}")
    return 0


def _cmd_self_test(args: argparse.Namespace) -> int:
    try:
        run_self_test = importlib.import_module(f"{__package__}.selftest").run_self_test
    except (ImportError, AttributeError) as exc:
        raise CLIError(f"The self-test isn't available in this copy of Athena ({exc}).") from None
    return int(run_self_test(quick=args.quick) or 0)


def _cmd_import(args: argparse.Namespace) -> int:
    lib = Library()
    failures = 0
    for name in args.files:
        path = Path(name).expanduser()
        if not path.is_file():
            _err(f"Couldn't import '{name}': there's no such file.")
            failures += 1
            continue
        try:
            _import(lib, path)
        except IngestError as exc:
            _err(f"Couldn't import '{path.name}': {exc}")
            failures += 1
    return 1 if failures else 0


def _cmd_list(args: argparse.Namespace) -> int:
    books = Library().list()
    if not books:
        print("Your library is empty. Add a file with: athena import FILE")
        return 0
    width = min(50, max(len(b.title) for b in books))
    for book in books:
        title = book.title if len(book.title) <= width else book.title[: width - 1] + "…"
        where = ""
        if book.progress is not None:
            where = f"  at {book.progress.section_id}, sentence {book.progress.utterance + 1}"
        print(f"{book.id}  {title:<{width}}  {book.source_type:<5} {_plural(book.section_count, 'section')}{where}")
    return 0


def _cmd_toc(args: argparse.Namespace) -> int:
    lib = Library()
    doc = resolve_book(lib, args.book)
    print(_book_line(doc.title, len(doc.sections), doc.page_count))
    id_width = max(len(s.id) for s in doc.sections)
    for section in doc.sections:
        indent = "  " * max(0, section.level - 1)
        pages = ""
        if section.page_start is not None:
            same = section.page_end in (None, section.page_start)
            pages = f"  (p. {section.page_start})" if same else f"  (pp. {section.page_start}-{section.page_end})"
        print(f"  {section.id:<{id_width}}  {indent}{section.title}{pages}")
    return 0


def _cmd_text(args: argparse.Namespace) -> int:
    lib = Library()
    doc = resolve_book(lib, args.book)
    sections = [doc.sections[_section_index(doc, args.section)]] if args.section else doc.sections
    options = config.load_settings().reading
    plan_section = _speech().plan_section if args.speech else None
    for n, section in enumerate(sections):
        if len(sections) > 1:
            if n:
                print()
            print(f"=== {section.title} [{section.id}] ===")
        if plan_section is not None:
            for utterance in plan_section(section, options):
                print(utterance.speech)
        else:
            for block in section.blocks:
                print(_display_block(block))
                print()
    return 0


def _start_position(lib: Library, doc: Document, section_id: str | None) -> tuple[int, int, bool]:
    """(section index, utterance, resumed) to start reading at."""
    if section_id:
        return _section_index(doc, section_id), 0, False
    progress = lib.get_progress(doc.id)
    if progress is not None:
        for i, section in enumerate(doc.sections):
            if section.id == progress.section_id:
                return i, progress.utterance, True
    return 0, 0, False


def _wait(future: Future[AudioClip]) -> AudioClip:
    """``future.result()`` in short waits: on Windows an endless wait doesn't notice Ctrl+C."""
    while True:
        try:
            return future.result(timeout=0.2)
        except FutureTimeout:
            continue


def _cmd_read(args: argparse.Namespace) -> int:
    lib = Library()
    doc = resolve_book(lib, args.book)
    settings = config.load_settings()
    engine = _choose_engine(args.engine, settings)
    # Same voice choice and cache key as the web reader, so sentences heard there play instantly here.
    voice = _pick_voice(engine, args.voice, settings)
    rate = _clamp_rate(args.rate if args.rate is not None else settings.speed)
    speech, audio = _speech(), _audio()
    cache = _audio_cache_class()(config.cache_dir())
    rate_key = _rate_key(rate)
    trim = getattr(audio, "trim_silence", None)

    def make(text: str) -> AudioClip:
        clip = engine.synthesize(text, voice, rate)
        if trim is not None and clip.mime == "audio/wav":
            try:
                clip = trim(clip)
            except Exception:  # an untrimmed clip is better than no audio
                pass
        return clip

    def synthesize(text: str) -> AudioClip:
        return cache.get_or_create((engine.name, voice or "", rate_key, text), lambda: make(text))

    start_section, start_utterance, resumed = _start_position(lib, doc, args.section)
    print(f"Reading \"{doc.title}\" with {engine.label}. Press Ctrl+C to stop.")
    if resumed:
        print("Continuing where you left off.")
    # Synthesize the next sentence while the current one plays, so there are no gaps.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="athena-read")
    try:
        for index in range(start_section, len(doc.sections)):
            section = doc.sections[index]
            plan = speech.plan_section(section, settings.reading)
            first = start_utterance if index == start_section else 0
            if not 0 <= first < len(plan):
                first = 0
            print(f"\n== {section.title} ==", flush=True)
            pending: Future[AudioClip] | None = pool.submit(synthesize, plan[first].speech) if plan else None
            for i in range(first, len(plan)):
                utterance = plan[i]
                assert pending is not None
                clip = _wait(pending)
                pending = pool.submit(synthesize, plan[i + 1].speech) if i + 1 < len(plan) else None
                lib.set_progress(doc.id, section.id, i)
                print(_utterance_text(section, utterance), flush=True)
                audio.play_clip(clip)
                if utterance.pause_after_ms > 0:
                    time.sleep(utterance.pause_after_ms / 1000 / max(rate, 0.1))
            if index + 1 < len(doc.sections):
                lib.set_progress(doc.id, doc.sections[index + 1].id, 0)
        print("\nYou've reached the end of the book.")
    except KeyboardInterrupt:
        print("\nStopped. Your place is saved; run the same command to continue.")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    lib = Library()
    doc = resolve_book(lib, args.book)
    section_ids: list[str] | None = None
    if args.section:
        section_ids = list(dict.fromkeys(args.section))
        for section_id in section_ids:
            _section_index(doc, section_id)
    settings = config.load_settings()
    engine = _choose_engine(args.engine, settings)
    voice = _pick_voice(engine, args.voice, settings)
    out_dir = Path(args.out).expanduser() if args.out else Path.cwd() / _safe_folder_name(doc.title)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CLIError(f"Couldn't create the folder {out_dir}: {exc.strerror or exc}") from None
    export_audiobook = _export_audiobook()
    count = len(section_ids) if section_ids else len(doc.sections)
    print(f"Saving {_plural(count, 'section')} of \"{doc.title}\" as {args.format.upper()} with {engine.label}...")
    cancel = threading.Event()
    bar = _ProgressBar()
    try:
        paths = export_audiobook(
            doc,
            engine=engine,
            voice=voice,
            options=settings.reading,
            out_dir=out_dir,
            section_ids=section_ids,
            fmt=args.format,
            rate=_clamp_rate(args.rate),
            progress=bar,
            cancel=cancel,
        )
    except KeyboardInterrupt:
        cancel.set()
        bar.finish()
        _err("Export cancelled.")
        return 130
    finally:
        bar.finish()
    print(f"Saved {_plural(len(paths), 'file')} in {out_dir}:")
    for path in paths:
        print(f"  {Path(path).name}")
    return 0


def _cmd_voices(args: argparse.Namespace) -> int:
    if args.voices_cmd == "catalog":
        return _voices_catalog()
    if args.voices_cmd == "download":
        return _voices_download(args.voice_id)
    settings = config.load_settings()
    tts = _tts()
    if args.engine:
        try:
            engines = [tts.get_engine(args.engine)]
        except KeyError:
            raise CLIError(f"There's no voice engine called '{args.engine}'.") from None
    else:
        engines = list(tts.all_engines())
    for engine in engines:
        header = f"{engine.name}: {engine.label}" + (" (online)" if engine.online else "")
        if engine.online and not settings.online_tts_allowed:
            print(f"{header}\n  {ONLINE_TTS_OFF}")
            continue
        try:
            available, reason = engine.is_available()
            voices = [v for v in engine.list_voices() if v.installed] if available else []
        except Exception as exc:
            available, reason, voices = False, str(exc), []
        if not available:
            print(f"{header}\n  Not available: {reason}")
            continue
        print(f"{header}  (ready)")
        if not voices:
            hint = "  (the browser picks its own voices)" if engine.name == "browser" else "  No voices installed."
            print(hint)
        for voice in voices:
            details = ", ".join(x for x in (voice.language, voice.gender) if x)
            print(f"  {voice.id}  {voice.name}" + (f"  ({details})" if details else ""))
    return 0


def _voices_catalog() -> int:
    catalog = _catalog()
    finder = getattr(catalog, "installed_catalog_ids", None)
    installed = set(finder()) if finder else {path.stem for path in catalog.installed_piper_voices()}
    recommended = getattr(catalog, "RECOMMENDED_VOICE", DEFAULT_VOICE_ID)
    print("Natural offline voices you can download:")
    width = max((len(v.id) for v in catalog.CATALOG), default=10)
    for voice in catalog.CATALOG:
        mark = "installed" if voice.id in installed else f"{voice.size_mb:.0f} MB"
        star = " (recommended)" if voice.id == recommended or getattr(voice, "recommended", False) else ""
        print(f"  {voice.id:<{width}}  {mark:>9}  {voice.name}{star}")
        if voice.description:
            print(f"  {'':<{width}}             {voice.description}")
    print("\nDownload one with: athena voices download VOICE_ID")
    return 0


def _voices_download(voice_id: str | None) -> int:
    catalog = _catalog()
    if not voice_id:
        raise CLIError("Say which voice to download, e.g.: athena voices download " + DEFAULT_VOICE_ID)
    check = getattr(catalog, "is_valid_voice_id", None)
    if voice_id not in {v.id for v in catalog.CATALOG} and not (check is not None and check(voice_id)):
        raise CLIError(
            f"'{voice_id}' isn't a voice you can download (expected something like {DEFAULT_VOICE_ID}). "
            "See the list with: athena voices catalog"
        )
    print(f"Downloading {voice_id}...")
    cancel = threading.Event()
    bar = _ProgressBar()
    try:
        path = catalog.download_piper_voice(voice_id, progress=bar, cancel=cancel)
    except KeyboardInterrupt:
        cancel.set()
        bar.finish()
        _err("Download cancelled.")
        return 130
    finally:
        bar.finish()
    print(f"Done. The voice is saved in {Path(path).parent}.")
    if config.load_settings().tts_engine in ("auto", "piper"):
        print("Athena will use Piper voices automatically.")
    return 0


def _cmd_ai_key(args: argparse.Namespace) -> int:
    if args.remove:
        config.update_settings({"anthropic_api_key": ""})
        print("Your Claude API key was removed from Athena's settings.")
        if os.environ.get("ANTHROPIC_API_KEY"):
            print("Note: the ANTHROPIC_API_KEY environment variable is still set, so a key is still available.")
        return 0
    try:
        key = getpass.getpass("Paste your Claude API key (it won't be shown): ").strip()
    except EOFError:
        key = ""
    if not key:
        raise CLIError("No key was entered, so nothing changed.")
    if any(ch.isspace() for ch in key) or not key.isprintable():
        raise CLIError("That key contains spaces or invisible characters. Copy it again and retry.")
    settings = config.update_settings({"anthropic_api_key": key})
    print("Saved your Claude API key in Athena's settings on this computer.")
    if not key.startswith("sk-"):
        print("Note: Claude API keys usually start with 'sk-'. If AI helpers fail, check the key.")
    if not settings.ai_enabled:
        print("AI study helpers are still turned off. Turn them on in the reader's Settings when you want them.")
    return 0


def _microsoft_store_python() -> bool:
    """True for Python from the Microsoft Store, which redirects writes to %LOCALAPPDATA%."""
    if sys.platform != "win32":
        return False
    places = (sys.prefix, sys.base_prefix, sys.executable)
    return any("\\windowsapps\\" in (place or "").lower().replace("/", "\\") for place in places)


def _cmd_doctor(args: argparse.Namespace) -> int:
    settings = config.load_settings()
    python_ok = sys.version_info >= (3, 10)
    print(f"Athena {__version__}")
    print(f"Python {platform.python_version()}" + ("" if python_ok else "  (too old: Athena needs 3.10 or newer)"))
    print(f"Data folder: {config.data_dir()}")
    if _microsoft_store_python():
        print(
            "  Warning: this Python comes from the Microsoft Store. Windows quietly keeps its copy of the data\n"
            "  folder inside the Store app's own storage (under %LOCALAPPDATA%\\Packages), so File Explorer\n"
            "  and the Athena app may not see your books, and uninstalling that Python deletes them.\n"
            "  Use the Athena app or Python from python.org instead."
        )
    try:
        book_count = len(Library().list())
    except OSError as exc:
        print(f"Library: couldn't be read ({exc})")
    else:
        print(f"Library: {_plural(book_count, 'book')}")

    print("\nVoices:")
    try:
        engines = list(_tts().all_engines())
    except Exception as exc:
        engines = []
        print(f"  Voice engines couldn't load: {exc}")
    for engine in engines:
        label = f"{engine.name} ({engine.label})"
        if engine.online:
            label += " [online]"
        try:
            available, reason = engine.is_available()
        except Exception as exc:
            available, reason = False, f"failed to start: {exc}"
        state = "ready" if available else f"not available: {reason}"
        if available and engine.online and not settings.online_tts_allowed:
            state = "installed, but online voices are turned off in Settings"
        elif available and reason:
            state += f" ({reason})"
        print(f"  {label}: {state}")
    if engines:
        try:
            chosen = _tts().resolve_engine(settings.tts_engine, online_allowed=settings.online_tts_allowed)
            print(f"  Reading with: {chosen.name}")
        except Exception as exc:
            print(f"  Couldn't choose a voice: {exc}")

    ocr_ok, ocr_reason = _ocr_status()
    ocr_state = "ready" if ocr_ok else "not available"
    print(f"\nScanned pages and photos (OCR): {ocr_state}" + (f" - {ocr_reason}" if ocr_reason else ""))

    key_state = "set" if settings.resolved_api_key() else "not set (run: athena ai-key)"
    print(f"AI study helpers: {'on' if settings.ai_enabled else 'off'}; API key {key_state}; model {settings.ai_model}")
    return 0


def _cmd_wipe(args: argparse.Namespace) -> int:
    if not args.yes:
        if not _isatty(sys.stdin):
            raise CLIError("To erase everything without being asked, run: athena wipe --yes")
        print(
            "This deletes your library, reading progress, bookmarks, saved audio, exports and settings\n"
            "(including your Claude API key). Downloaded voices are kept."
        )
        try:
            answer = input("Type DELETE to confirm: ")
        except EOFError:
            answer = ""
        if answer.strip() != "DELETE":
            print("Nothing was deleted.")
            return 1
    wipe_user_data()
    print("All your Athena data was erased (downloaded voices were kept).")
    return 0


# -- argument parsing


def _positive_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("the port must be a number") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("the port must be between 1 and 65535")
    return port


def _rate(value: str) -> float:
    try:
        rate = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("the rate must be a number like 1.25") from None
    if not 0.5 <= rate <= 3.0:
        raise argparse.ArgumentTypeError("the rate must be between 0.5 and 3.0")
    return rate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="athena",
        description="Athena: a private study reader that reads your textbooks aloud.",
        epilog="Run 'athena' with no command to open the reader in your browser.",
    )
    parser.add_argument("--version", action="version", version=f"Athena {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("serve", help="open the reader in your browser (the default)")
    p.add_argument("--port", type=_positive_port, default=8765, help="port to use (default 8765)")
    p.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    p.add_argument("--host", default=None, help="address to listen on (default 127.0.0.1, this computer only)")
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("app", help="open the reader in its own window")
    p.add_argument("--port", type=_positive_port, default=8765, help="port to use (default 8765)")
    p.set_defaults(func=_cmd_app)

    p = sub.add_parser("shortcut", help="add Athena shortcuts to the desktop and Start menu")
    p.add_argument("--no-start-menu", action="store_true", help="only create the desktop shortcut")
    p.set_defaults(func=_cmd_shortcut)

    p = sub.add_parser("self-test", help="check that reading, voices and the reader all work")
    p.add_argument("--quick", action="store_true", help="skip the slower checks")
    p.set_defaults(func=_cmd_self_test)

    p = sub.add_parser("import", help="add files to your library")
    p.add_argument("files", nargs="+", metavar="FILE")
    p.set_defaults(func=_cmd_import)

    p = sub.add_parser("list", help="list the books in your library")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("toc", help="show a book's sections")
    p.add_argument("book", metavar="BOOK")
    p.set_defaults(func=_cmd_toc)

    p = sub.add_parser("text", help="print a book's text")
    p.add_argument("book", metavar="BOOK|FILE")
    p.add_argument("--section", metavar="ID", help="only this section (see 'athena toc')")
    p.add_argument("--speech", action="store_true", help="print what the voice will say instead")
    p.set_defaults(func=_cmd_text)

    p = sub.add_parser("read", help="read a book aloud here in the terminal")
    p.add_argument("book", metavar="BOOK|FILE")
    p.add_argument("--section", metavar="ID", help="start at this section instead of where you left off")
    p.add_argument("--engine", metavar="E", help="voice engine: auto, piper, system, edge")
    p.add_argument("--voice", metavar="V", help="voice id (see 'athena voices')")
    p.add_argument("--rate", type=_rate, metavar="R", help="speed, e.g. 1.25 (default: your reader speed)")
    p.set_defaults(func=_cmd_read)

    p = sub.add_parser("export", help="save a book as MP3 or WAV audio files")
    p.add_argument("book", metavar="BOOK")
    p.add_argument("--section", metavar="ID", action="append", help="only this section (repeat for more)")
    p.add_argument("--format", choices=("mp3", "wav"), default="mp3")
    p.add_argument("--out", metavar="DIR", help="folder for the files (default: a new folder named after the book)")
    p.add_argument("--engine", metavar="E", help="voice engine: auto, piper, system, edge")
    p.add_argument("--voice", metavar="V", help="voice id (see 'athena voices')")
    p.add_argument("--rate", type=_rate, default=1.0, metavar="R", help="speed, e.g. 1.25 (default 1.0)")
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("voices", help="list voices, or download natural offline voices")
    p.add_argument("--engine", metavar="E", help="only this engine")
    voices_sub = p.add_subparsers(dest="voices_cmd", metavar="ACTION")
    voices_sub.add_parser("catalog", help="voices you can download")
    q = voices_sub.add_parser("download", help="download a voice")
    q.add_argument("voice_id", metavar="VOICE_ID")
    p.set_defaults(func=_cmd_voices, voices_cmd=None)

    p = sub.add_parser("ai-key", help="save (or remove) your Claude API key for the study helpers")
    p.add_argument("--remove", action="store_true", help="remove the saved key")
    p.set_defaults(func=_cmd_ai_key)

    p = sub.add_parser("doctor", help="check what's installed and working")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("wipe", help="erase all your Athena data (keeps downloaded voices)")
    p.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    p.set_defaults(func=_cmd_wipe)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``athena`` command; returns the exit code."""
    _make_output_safe()
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help, --version or a usage error (already printed)
        return exc.code if isinstance(exc.code, int) else 2
    if args.command is None:
        args = parser.parse_args(["serve"])
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        _err("\nStopped.")
        return 130
    except CLIError as exc:
        _err(f"Error: {exc}")
        return 1
    except BrokenPipeError:  # output piped into a program that stopped reading, e.g. `| head`
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())  # so the final flush at exit doesn't fail again
        except (OSError, ValueError):
            pass
        return 0
    except OSError as exc:
        detail = f"{exc.strerror}: {exc.filename}" if exc.strerror and exc.filename else str(exc)
        _err(f"Error: {detail}")
        return 1
    except Exception as exc:
        if is_friendly_error(exc):
            _err(f"Error: {exc}")
            return 1
        _err(f"Something went wrong: {exc or type(exc).__name__}")
        _err("Details for a bug report:")
        traceback.print_exc()
        return 1
