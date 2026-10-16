# Windows voice and smoothness check

This guide is for a Claude Code session running on the reader's own Windows laptop.
Athena was built and tested in the cloud, on Linux and on GitHub's Windows Server
machines. Only real hardware can answer what's left: how smooth the voices sound on
this computer, and whether anything Windows-specific goes wrong. Work through the
steps in order and fix what you find.

## Ground rules: this is someone else's private computer

- Work only inside this repository folder. Don't open, list or search anything else on
  the computer: documents, downloads, browser data, other apps.
- Use test material only: the generated samples below, or public-domain text. Never
  open the owner's own books or files unless the person at the keyboard hands you a
  specific file for testing.
- Keep all test data inside the repo by setting `DESTINY_HOME` (step 1). Don't touch
  the real Athena data folder (`%LOCALAPPDATA%\Athena`).
- Ask before installing anything outside this folder, and before any download over
  about 50 MB. The voices in step 1 are about 180 MB, so ask first.
- Online features stay off. Turn on the Microsoft online voices (they send text to
  Microsoft) only if the person agrees, and only with the test material. Leave the
  Claude study helper off.
- Don't put personal details (user name, computer name, folder paths that include the
  user name) in commits or reports.

## 1. Set up an isolated test copy

Run these in PowerShell, in the repository folder:

```powershell
py -3 -m venv .venv            # Python 3.10+; if `py` is missing, ask the person to install Python
.venv\Scripts\python -m pip install -e ".[all,dev]"
$env:DESTINY_HOME = "$PWD\.athena-test-data"   # set it again in every new terminal
.venv\Scripts\python -m destiny_bot voices download en_US-lessac-medium
.venv\Scripts\python -m destiny_bot voices download kokoro-v1.0
.venv\Scripts\python -m destiny_bot doctor
```

`.athena-test-data` is already in `.gitignore`, so test books and settings never get committed.

## 2. Measure this computer

```powershell
.venv\Scripts\python -m destiny_bot self-test
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
powercfg /getactivescheme
```

Write down these values from `.athena-test-data\logs\self-test.txt`:

- **Kokoro:** first-sentence latency, cold and warm, and RTF while reading.
- **Piper:** the same three numbers.
- **System voice:** whether it speaks, and which SAPI voices exist.

For comparison, GitHub's 4-vCPU Windows machines measured:

- **Kokoro:** 2.7–4.9 s cold, 1.5–2.0 s warm, RTF 0.38–0.48.
- **Piper:** 0.17–0.20 s warm, RTF about 0.05.

To keep up at 2× speed, Kokoro needs an RTF below 0.5. If this computer is slower than
that, check how the reader handles it (step 4) and whether Piper should be its default.

Run the measurements twice: once plugged in, and once on battery with the
power-saver plan. Laptops throttle the CPU on battery.

## 3. Run the test suites on Windows

```powershell
.venv\Scripts\python -m pytest -m "not e2e" -q
.venv\Scripts\python -m pytest tests/e2e -m e2e -q
```

The end-to-end tests drive the real reader in a browser. Read
`tests/e2e/conftest.py` to see how they find one. Playwright can use the installed
Microsoft Edge (`channel="msedge"`), so you shouldn't need to download a browser. If
the suite only supports a downloaded Chromium, ask before running
`playwright install chromium` (about 150 MB), or add Edge support.

## 4. Listen, together with the person at the keyboard

The point of this step is how it sounds, which no automated test can judge.

```powershell
.venv\Scripts\python -m destiny_bot app
```

1. Make a realistic sample. `tests/pdf_factory.py` builds textbook-like PDFs, and
   `tests/test_ingest_pdf.py` shows how to call it. A public-domain chapter pasted
   through **Paste text** also works. Import it.
2. Play it with each voice. Settings → Voice lists Kokoro, Piper and the Windows system
   voice. Try speeds 1×, 1.5×, 2× and 3×. For each, ask the person to note:
   - the delay between pressing ▶ and the first word;
   - gaps, stutters, clicks or pops between sentences, or pauses that feel too long or
     too short;
   - robotic or metallic sound, especially at 2× and 3×;
   - mispronounced words (write down the exact text);
   - whether the highlighted sentence matches what is being spoken.
3. The reader has a timing/debug hook for measuring gaps, stalls and highlight drift.
   Read the top of `src/destiny_bot/web/app.js` and `tests/e2e/harness.py` to see how to
   switch it on. Capture its numbers alongside what the person hears.
4. Try the real-world disturbances:
   - Bluetooth headphones: connect and disconnect them mid-sentence.
   - Close the lid or let the computer sleep during playback, then resume.
   - Switch the audio output device.
   - Use the media keys on the keyboard.
   - Read for 10 minutes or more at 2×.
   - Save a chapter as audio, then play the MP3 in Windows Media Player.

## 5. Fix what you find

Where things live:

- **Kokoro and Piper engines:** `src/destiny_bot/tts/kokoro_engine.py` and `piper_engine.py`. Threads,
  model loading and speed are handled here.
- **Windows system voice:** `src/destiny_bot/tts/system_engine.py`. It uses SAPI 5 through
  comtypes, with PowerShell System.Speech as the fallback.
- **Playback scheduling and highlighting:** the Web Audio scheduler in `src/destiny_bot/web/app.js`.
  This covers prefetch depth, pauses and stall handling.
- **What gets said:** `src/destiny_bot/speech/normalize.py` (pronunciation) and `speech/plan.py`
  (pauses and sentence chunks).
- **Window, shortcuts and single-instance:** `src/destiny_bot/desktop.py`.

For every fix, write a regression test, run both suites, then commit with a clear
message and push to the branch `claude/ai-textbook-tts-bot-7m0t72`.

## 6. Report and clean up

1. Write `docs/windows-voice-report.md` with:
   - the hardware summary (CPU model, cores, RAM, and nothing personal);
   - the measurements from steps 2 and 4;
   - what the person heard;
   - what you fixed, and what's still open.
   Commit and push it.
2. Delete `.athena-test-data`. Ask whether to keep `.venv`, which is only needed for
   development.
3. Remind the person that they can sign out of Claude, or uninstall Claude Code, when
   they're done. Nothing else needs undoing: you only worked in this folder.
