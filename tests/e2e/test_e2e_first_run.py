"""The first-run experience for a non-technical user, in the real UI.

Empty library, importing a big PDF with visible progress, friendly errors, the
default voice, keyboard and screen-reader basics, and layouts at common sizes.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from .harness import Server, book_text

pytestmark = pytest.mark.e2e

# The most common cheap Windows laptop, 1920x1080 at 125% zoom, and a phone.
SIZES = {
    "laptop-1366": {"viewport": {"width": 1366, "height": 768}},
    "fullhd-125": {"viewport": {"width": 1536, "height": 864}, "device_scale_factor": 1.25},
    "phone-390": {"viewport": {"width": 390, "height": 844}, "device_scale_factor": 2, "is_mobile": True, "has_touch": True},
}
TECHNICAL = re.compile(r"Traceback|Exception|Errno|errno|\bat 0x|\.py\b|NoneType|KeyError|ValueError|stack", re.I)


@pytest.fixture(scope="module")
def fresh(tmp_path_factory, chromium_path):
    """A brand-new Athena (empty library) with Kokoro and Piper installed."""
    srv = Server(tmp_path_factory.mktemp("first-run-home"))
    yield srv
    srv.stop()


def no_horizontal_scroll(app) -> bool:
    return app.page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")


def visible_in_viewport(app, selector: str) -> bool:
    """The element is on screen and nothing covers its middle."""
    return app.page.evaluate("""(sel) => {
      const node = document.querySelector(sel);
      if (!node) return false;
      const r = node.getBoundingClientRect();
      if (r.width === 0 || r.bottom > window.innerHeight + 1 || r.right > window.innerWidth + 1 || r.top < 0) return false;
      const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
      return !!hit && (hit === node || node.contains(hit));
    }""", selector)


def make_pdf(path: Path, pages: int) -> Path:
    import pymupdf

    doc = pymupdf.open()
    for n in range(1, pages + 1):
        page = doc.new_page()
        if n % 20 == 1:
            page.insert_text((72, 90), f"Chapter {n // 20 + 1}: Cells and energy", fontsize=20, fontname="hebo")
        body = " ".join(f"Page {n} sentence {k} explains how cells turn food into energy." for k in range(1, 9))
        page.insert_textbox(pymupdf.Rect(72, 120, 540, 720), body, fontsize=11, fontname="helv")
    doc.save(path)
    return path


def test_empty_library_says_what_to_do_and_kokoro_is_preselected(fresh, open_app):
    for name, context in SIZES.items():
        app = open_app(fresh.base, **context)
        app.wait("document.querySelector('#welcome-title')")
        assert app.page.text_content("#welcome-title") == "Welcome to Athena"
        lead = app.page.text_content(".welcome-lead")
        assert "read them aloud" in lead
        assert app.page.is_visible(".welcome-actions >> text=Add material")
        assert app.page.is_visible(".welcome-actions >> text=Paste text")
        assert no_horizontal_scroll(app), name
        assert visible_in_viewport(app, "#btn-play"), name
        # The best voice is first and already chosen, with a friendly name.
        picker = app.page.eval_on_selector("#voice-select", """s => ({
          value: s.value, text: s.selectedOptions[0].textContent,
          first: s.querySelector('optgroup').label, firstVoice: s.querySelector('optgroup option').textContent })""")
        assert picker == {"value": "kokoro::af_heart", "text": "Heart (US English, female)",
                          "first": "Kokoro (most natural, offline)", "firstVoice": "Heart (US English, female)"}, name
        assert app.page.get_attribute("#btn-play", "aria-label") == "Play"
        assert app.page.is_disabled("#btn-play")
        app.page.screenshot(path=str(Path(fresh.home) / f"empty-{name}.png"))
        app.check_no_errors()
    # On a laptop the empty library says what to do as well.
    app = open_app(fresh.base)
    app.wait("document.querySelector('#welcome-title')")
    assert app.page.is_visible("#library-empty")
    assert "Add a textbook" in app.page.text_content("#library-empty")

    # Settings explain the choice in plain words.
    app.page.click("#btn-settings")
    app.wait("document.querySelector('#engine-list .engine-card')")
    auto = app.page.locator("#engine-list .engine-card", has=app.page.locator("#engine-auto"))
    assert app.page.is_checked("#engine-auto")
    assert "best voice available" in auto.text_content() and "Kokoro" in auto.text_content()
    assert app.page.eval_on_selector("#set-voice", "s => s.selectedOptions[0].textContent") == "Heart (US English, female)"
    app.page.keyboard.press("Escape")
    app.check_no_errors()


def test_default_voice_is_piper_lessac_without_kokoro(tmp_path_factory, chromium_path, open_app):
    srv = Server(tmp_path_factory.mktemp("piper-only-home"), kokoro=False)
    try:
        app = open_app(srv.base)
        app.wait("document.querySelector('#welcome-title')")
        assert app.page.eval_on_selector("#voice-select", "s => s.value") == "piper::en_US-lessac-medium"
        assert app.page.eval_on_selector("#voice-select", "s => s.selectedOptions[0].textContent") == "Lessac (US English)"
        app.check_no_errors()
    finally:
        srv.stop()


def test_import_shows_progress_and_errors_are_friendly(fresh, open_app, tmp_path):
    app = open_app(fresh.base)
    app.wait("document.querySelector('#welcome-title')")
    pdf = make_pdf(tmp_path / "Biology textbook.pdf", 900)
    app.page.set_input_files("#file-input", str(pdf))
    seen = set()
    deadline = time.time() + 120
    while time.time() < deadline:
        msg = app.page.evaluate("""(() => { const t = document.querySelector('#tasks .task');
          return t ? [t.querySelector('.task-msg').textContent, t.querySelector('[role=progressbar]').getAttribute('aria-valuenow')] : null; })()""")
        if msg:
            seen.add(tuple(msg))
        if app.page.query_selector(".article .section-title"):
            break
        time.sleep(0.05)
    messages = [m for m, _ in seen]
    assert any(re.search(r"page \d+ of 900", m) for m in messages), messages
    assert any(p not in (None, "0", "100") for _, p in seen), seen
    app.wait("[...document.querySelectorAll('.toast')].some((t) => t.textContent.includes('Added'))", timeout=10)
    assert "Chapter 1" in app.page.text_content(".article .section-title")

    # A damaged PDF and an unsupported file give plain-language messages.
    broken = tmp_path / "lecture notes.pdf"
    broken.write_bytes(b"%PDF-1.4\n this is not really a pdf")
    program = tmp_path / "setup.exe"
    program.write_bytes(b"MZ\x90\x00 not a document")
    for path, expected in ((broken, "PDF"), (program, ".exe")):
        app.page.set_input_files("#file-input", str(path))
        failed = app.wait(f"""(() => {{ const t = [...document.querySelectorAll('#tasks .task.is-error')]
          .find((n) => n.textContent.includes({path.name!r})); return t && t.querySelector('.task-msg').textContent; }})()""",
                          timeout=30)
        assert expected in failed and not TECHNICAL.search(failed), failed
    app.check_no_errors(allowed=("status of 400",))


def test_keyboard_and_screen_reader_basics(fresh, open_app):
    fresh.settings(tts_engine="piper", speed=1.0)
    book = fresh.paste("Keyboard check", book_text(1, 2, 3))
    app = open_app(fresh.base, book_id=book["id"])
    app.open_book()
    page = app.page
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.textContent.trim()") == "Skip to the text"
    page.evaluate("document.activeElement.blur()")
    page.keyboard.press("Space")
    app.wait("athenaDebug.state().playing")
    assert page.get_attribute("#btn-play", "aria-label") == "Pause"
    app.wait("document.querySelector('#live').textContent === 'Playing'", timeout=5)
    app.wait("athenaDebug.events.some((e) => e.type === 'highlight')", timeout=30)
    before = app.state()["index"]
    page.keyboard.press("ArrowRight")
    app.wait(f"athenaDebug.state().index === {before + 1}", timeout=5)
    page.keyboard.press("Space")
    app.wait("!athenaDebug.state().playing")
    page.keyboard.press("Shift+?")
    assert page.is_visible("#shortcuts-dialog")
    page.keyboard.press("Escape")
    assert not page.is_visible("#shortcuts-dialog")
    # Every control in the player has a name a screen reader can say.
    unnamed = page.evaluate("""[...document.querySelectorAll('#player button, #player input, #player select')]
      .filter((n) => !(n.getAttribute('aria-label') || n.labels?.length || n.textContent.trim()))
      .map((n) => n.id || n.outerHTML.slice(0, 60))""")
    assert unnamed == []
    assert page.get_attribute("#now-status", "aria-live") == "off"  # sentence counter doesn't chatter
    app.check_no_errors()


def test_layout_with_a_book_open_at_common_sizes(fresh, open_app):
    fresh.settings(tts_engine="piper", speed=1.0)
    book = fresh.paste("Layout check", book_text(2, 3, 4))
    for name, context in SIZES.items():
        app = open_app(fresh.base, book_id=book["id"], **context)
        app.open_book()
        assert no_horizontal_scroll(app), name
        for selector in ("#btn-play", "#speed", "#voice-select", ".article .section-title"):
            assert visible_in_viewport(app, selector), (name, selector)
        # The reading column is wide enough to read comfortably.
        width = app.page.evaluate("document.querySelector('.article').getBoundingClientRect().width")
        assert width >= min(340, context["viewport"]["width"] - 40), (name, width)
        app.page.click("#btn-settings")
        app.wait("document.querySelector('#engine-list .engine-card')")
        assert no_horizontal_scroll(app), name
        assert visible_in_viewport(app, "#settings-dialog [data-close]"), name
        app.page.screenshot(path=str(Path(fresh.home) / f"settings-{name}.png"))
        app.page.keyboard.press("Escape")
        app.check_no_errors()
