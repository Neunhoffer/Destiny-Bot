"""Playback smoothness in the real UI with the real Kokoro and Piper voices.

Timing comes from the app's own ``?debug`` hooks: each scheduled sentence records
when it was planned to start and when it really starts on the audio clock, so a
"stall" (the next sentence not ready in time) shows up as ``start > planned``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time

import pytest

from .harness import book_text, max_overlap, pause_seconds, sound_time

pytestmark = pytest.mark.e2e

SHORT = "short and clear for listening"


def utterances(server, book_id: str, section: str = "s0") -> list[dict]:
    return server.api("GET", f"/api/books/{book_id}/sections/{section}")["utterances"]


def late(event: dict) -> float:
    return event["start"] - event["planned"]


def test_kokoro_starts_at_once_fetches_in_order_and_keeps_up(server, open_app):
    """Defaults to Kokoro; Play is instant after opening; one request at a time, next-needed first."""
    server.settings(tts_engine="auto", speed=1.0)
    book = server.paste("Kokoro smoothness", book_text(2, 3, 5, words=SHORT))
    speech = [u["speech"] for u in utterances(server, book["id"])]
    app = open_app(server.base, book_id=book["id"])
    app.open_book()
    assert app.state()["engine"] == "kokoro"

    # Opening the book gets the first sentences ready before Play is pressed.
    app.wait("athenaDebug.state().clipBlobs >= 2", timeout=90)
    assert [r["text"] for r in app.tts_log()] == speech[:2]
    pressed = app.now()
    app.page.click("#btn-play")
    first = app.wait("athenaDebug.events.find((e) => e.type === 'schedule')", timeout=30)
    assert first["index"] == 0
    assert sound_time(first) - pressed < 1500, f"first sound after {sound_time(first) - pressed:.0f} ms ({first})"

    # Change the speed in the middle of the paragraph, then listen at 2x.
    app.wait("athenaDebug.state().index >= 2", timeout=60)
    app.set_speed(2.0)
    app.wait("athenaDebug.events.some((e) => e.type === 'rate')", timeout=5)
    time.sleep(9)
    state = app.state()
    rtf = state["rtf"]["kokoro"]
    assert state["lookahead"] > 40, "a slow voice should be fetched further ahead"
    assert state["plannedAhead"] > 3, state
    rate_at = app.events("rate")[0]["perf"]
    after = [e for e in app.events("schedule") if e["perf"] > rate_at]
    assert after, "nothing was read after the speed change"
    if rtf < 0.7:  # this computer keeps up with Kokoro at 2x, so nothing may stall
        assert max(late(e) for e in after) < 0.05, ([round(late(e), 2) for e in after], rtf)

    # Requests never overlap, and while reading straight on they follow the text.
    log = app.tts_log()
    assert max_overlap(log) == 1
    assert not any(r["normalize"] for r in log), "reading-plan sentences are already normalized"
    order = [speech.index(r["text"]) for r in log if r["text"] in speech]
    assert order[:6] == sorted(order[:6]), order

    # Seek somewhere not fetched yet: that sentence is requested first and heard right after it arrives.
    app.page.click("#btn-play")  # pause
    app.set_speed(1.25)
    time.sleep(0.6)
    target = 9
    count = len(app.tts_log())
    clicked = app.now()
    app.page.click(f".article [data-u='{target}']")
    heard = app.wait(f"athenaDebug.events.find((e) => e.type === 'schedule' && e.perf > {clicked})", timeout=60)
    assert heard["index"] == target
    issued = app.tts_log()[count:]
    assert issued[0]["text"] == speech[target], [r["text"][:30] for r in issued]
    waited_for_voice = issued[0]["t1"] - clicked
    extra = sound_time(heard) - clicked - waited_for_voice - heard["hold"] * 1000
    assert extra < 400, f"{extra:.0f} ms between the clip arriving and its sound"
    app.wait(f"document.querySelector(\".article [data-u='{target}']\").classList.contains('is-current')", timeout=10)
    app.check_no_errors()


def test_piper_pauses_match_the_plan_at_every_speed(server, open_app):
    """Gaps between sentences are exactly the planned pauses at 1x, 1.5x, 2x and 3x, changed mid-paragraph."""
    server.settings(tts_engine="piper", speed=1.0)
    book = server.paste("Piper speeds", book_text(1, 5, 6, words=SHORT))
    plan = utterances(server, book["id"])
    app = open_app(server.base, book_id=book["id"])
    app.open_book()
    app.page.click("#btn-play")
    app.wait("athenaDebug.state().index >= 1", timeout=30)
    for speed in (1.5, 2.0, 3.0):
        time.sleep(3.5)
        app.set_speed(speed)
    time.sleep(3.5)
    app.page.click("#btn-play")

    changes = [(1.0, 0.0)] + [(e["rate"], e["perf"]) for e in app.events("rate")]
    assert [r for r, _ in changes] == [1.0, 1.5, 2.0, 3.0]

    def rate_at(perf: float) -> float:
        return [r for r, t in changes if t <= perf][-1]

    scheduled = app.events("schedule")
    assert len(scheduled) >= 12
    firsts_after_change = {next(e["perf"] for e in scheduled if e["perf"] > t) for _, t in changes[1:]}
    per_char: dict[float, list[float]] = {}
    for e in scheduled[1:]:
        rate = rate_at(e["perf"])
        just_changed = e["perf"] in firsts_after_change
        assert late(e) < (1.0 if just_changed else 0.02), (e, late(e))
        assert abs(e["pause"] - pause_seconds(plan[e["index"]], rate)) < 0.002, (e, rate)
        per_char.setdefault(rate, []).append((e["end"] - e["start"]) / len(plan[e["index"]]["speech"]))
    slow, fast = per_char[1.0], per_char[3.0]
    assert sum(fast) / len(fast) < 0.5 * sum(slow) / len(slow), "3x speech should be much shorter"
    # The highlight moves when a sentence is heard, within a few animation frames.
    drift = [e["heard"] - e["start"] for e in app.events("highlight")]
    assert drift and min(drift) > -0.01 and max(drift) < 0.15, drift
    assert max_overlap(app.tts_log()) == 1
    app.check_no_errors()


def test_seek_pause_resume_next_section_and_voice_switch(server, open_app):
    server.settings(tts_engine="piper", speed=1.5)
    book = server.paste("Moving around", book_text(2, 2, 4))
    first = utterances(server, book["id"], "s0")
    app = open_app(server.base, book_id=book["id"])
    app.open_book()
    app.page.click("#btn-play")
    app.wait("athenaDebug.state().index >= 1", timeout=20)

    # Click a sentence: reading jumps there.
    clicked = app.now()
    app.page.click(".article [data-u='5']")
    jumped = app.wait(f"athenaDebug.events.find((e) => e.type === 'schedule' && e.perf > {clicked})", timeout=20)
    assert jumped["index"] == 5
    app.wait("athenaDebug.state().index === 5", timeout=10)

    # Pause holds the place; resume carries on from the same sentence.
    app.page.click("#btn-play")
    assert app.page.get_attribute("#btn-play", "aria-label") == "Play"
    app.wait("athenaDebug.state().ctxState === 'suspended'", timeout=5)
    held = app.state()["index"]
    time.sleep(1.2)
    assert app.state()["index"] == held
    app.page.click("#btn-play")
    app.wait("athenaDebug.state().ctxState === 'running'", timeout=5)
    app.wait(f"athenaDebug.state().index > {held}", timeout=20)

    # Reading flows into the next section on time, and the page follows it.
    last = len(first) - 1
    app.page.click(f".article [data-u='{last}']")
    app.wait("athenaDebug.state().section === 's1'", timeout=30)
    crossing = next(e for e in app.events("schedule") if e["section"] == "s1" and e["index"] == 0)
    assert late(crossing) < 0.05, crossing
    assert "Chapter 2" in app.page.text_content("#now-status")
    assert app.page.get_attribute("#toc [data-sid='s1']", "aria-current") == "location"

    # Switching voice mid-sentence restarts that sentence with the new voice.
    at = app.state()["index"]
    switched = app.now()
    app.page.select_option("#voice-select", "kokoro::af_heart")
    start = app.wait(f"athenaDebug.events.find((e) => e.type === 'start' && e.perf > {switched})", timeout=10)
    assert start["engine"] == "kokoro" and start["index"] in (at, at + 1)
    app.wait(f"athenaDebug.events.find((e) => e.type === 'schedule' && e.perf > {switched})", timeout=60)
    assert server.api("GET", "/api/settings")["tts_engine"] == "kokoro"
    app.check_no_errors()


def test_repeated_sentences_play_through(server, open_app):
    """Books repeat sentences: each clip is fetched once and the page never freezes."""
    server.settings(tts_engine="piper", speed=2.0)
    steps = ["Lab steps.", "Mix the solution.", "Mix the solution.", "Wait one minute.", "Mix the solution.", "Lab steps."]
    book = server.paste("Repeats", " ".join(steps) + "\n\nWrite down the colour.")
    plan = utterances(server, book["id"])
    start = next(u["index"] for u in plan if u["speech"] == "Mix the solution.")
    assert plan[start + 1]["speech"] == plan[start]["speech"]
    server.api("PUT", f"/api/books/{book['id']}/progress", {"section_id": "s0", "utterance": start})
    app = open_app(server.base, book_id=book["id"])
    app.open_book()
    # The warm-up asks for the current and next sentence: here the same text.
    app.wait("window.__ttsLog.length >= 1 && window.__ttsLog.every((r) => r.t1)", timeout=60)
    time.sleep(0.5)
    app.page.click("#btn-play", timeout=10000)
    app.wait("athenaDebug.state().playing", timeout=10)
    app.wait("!athenaDebug.state().playing", timeout=60)  # reached the end
    heard = [e["index"] for e in app.events("schedule")]
    assert heard == list(range(start, len(plan))), heard
    texts = [r["text"] for r in app.tts_log()]
    assert len(set(texts)) == len(texts), texts
    app.check_no_errors()


def test_slow_voice_shows_loading_and_offers_the_faster_voice(server, open_app):
    """When Kokoro can't keep up, she sees "Loading the voice…" and gets a one-click switch to Piper."""
    server.settings(tts_engine="kokoro", speed=1.0, voices={"kokoro": "af_heart"})
    book = server.paste("Slow computer", " ".join(f"Step {n} is next." for n in range(1, 40)))
    app = open_app(server.base, book_id=book["id"])
    app.open_book()
    app.js("(athenaDebug.slowTts = 1500, athenaDebug.fastForward = 2)")  # pretend synthesis is slow
    app.page.click("#btn-play")
    app.wait("document.body.classList.contains('is-loading-voice')", timeout=30)
    assert app.page.text_content("#now-status").startswith("Loading the voice")
    assert app.page.get_attribute("#btn-play", "aria-busy") == "true"
    app.wait("document.querySelector('.toast-offer')", timeout=90)
    offer = app.page.text_content(".toast-offer")
    assert "a bit slow for this voice" in offer and "faster natural voice" in offer
    app.js("(athenaDebug.slowTts = 0)")
    app.page.click(".toast-offer >> text=Switch voice")
    app.wait("athenaDebug.state().engine === 'piper' && athenaDebug.state().playing", timeout=10)
    assert server.api("GET", "/api/settings")["tts_engine"] == "piper"
    assert app.page.eval_on_selector("#voice-select", "s => s.value") == "piper::en_US-lessac-medium"
    assert app.page.evaluate("localStorage.getItem('destiny.slowVoiceOffer')") == '"switched"'
    app.check_no_errors()


def test_long_session_stays_bounded_and_resumes_exactly(server, open_app):
    """Many sections at 2x (fast-forwarded): memory stays bounded; reopening resumes on the same sentence."""
    server.settings(tts_engine="piper", speed=2.0)
    book = server.paste("Long session", book_text(6, 2, 5, words=SHORT))
    app = open_app(server.base, book_id=book["id"])
    app.open_book()
    app.js("(athenaDebug.fastForward = 6)")
    app.page.click("#btn-play")
    samples = []
    deadline = time.time() + 75
    while len(app.events("schedule")) < 36 and time.time() < deadline:
        time.sleep(2)
        samples.append(app.state())
    assert int(samples[-1]["section"][1:]) >= 2, "should have read through several sections"
    for st in samples:
        assert st["clipEntries"] <= 60 and st["objectUrls"] == 0, st
        assert st["scheduled"] <= 5 and st["sectionsCached"] <= 8, st
    if samples[0]["heapBytes"]:
        grew = samples[-1]["heapBytes"] - samples[min(2, len(samples) - 1)]["heapBytes"]
        assert grew < 8_000_000, f"heap grew {grew} bytes"
    assert app.page.evaluate("document.querySelectorAll('.article').length") == 1

    # Pause, close the window, open it again: the same sentence is current.
    app.page.click("#btn-play")
    time.sleep(0.5)
    where = app.state()
    app.page.close(run_before_unload=True)
    time.sleep(0.5)
    again = open_app(server.base, book_id=book["id"])
    again.open_book()
    st = again.state()
    assert (st["section"], st["index"]) == (where["section"], where["index"])
    assert again.page.text_content(".article .is-current")

    # Closing while reading also keeps the place (saved as the page goes away).
    again.page.click("#btn-play")
    again.wait("athenaDebug.events.filter((e) => e.type === 'highlight').length >= 3", timeout=30)
    seen = again.state()
    again.page.close(run_before_unload=True)
    time.sleep(0.5)
    third = open_app(server.base, book_id=book["id"])
    third.open_book()
    st3 = third.state()
    key = lambda s: (int(s["section"][1:]), s["index"])
    assert key(seen) <= key(st3) <= (key(seen)[0], key(seen)[1] + 3) or key(st3)[0] == key(seen)[0] + 1, (seen, st3)
    app.check_no_errors()


def test_export_mp3_with_kokoro_plays(server, open_app, tmp_path):
    server.settings(tts_engine="kokoro", speed=1.0)
    book = server.paste("Export check", "# Short chapter\n\nThis chapter is saved as audio. It has two sentences.")
    app = open_app(server.base, book_id=book["id"])
    app.open_book()
    app.page.click("#btn-export")
    assert app.page.eval_on_selector("#export-voice", "s => s.value") == "kokoro::af_heart"
    assert app.page.is_checked("input[name='export-format'][value='mp3']")
    app.page.click("#export-start")
    app.wait("document.querySelector('#export-files .file-link')", timeout=120)
    link = app.page.locator("#export-files .file-link", has_text=re.compile(r"\.mp3"))
    with app.page.expect_download() as download:
        link.first.click()
    path = tmp_path / download.value.suggested_filename
    download.value.save_as(path)
    data = path.read_bytes()
    assert data[:3] == b"ID3" or (data[0] == 0xFF and data[1] & 0xE0 == 0xE0), data[:4]
    if shutil.which("ffprobe"):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name:format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        assert probe[0] == "mp3" and float(probe[-1]) > 2.0, probe
    app.check_no_errors()
