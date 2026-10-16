# Athena 🦉📖🔊

**A private study reader that reads your textbooks and class materials out loud.**

Open a PDF textbook, an e-book, Word notes, lecture slides, or a photo of a handout.
Athena cleans the text up for listening:

- It skips page numbers, running headers, and citation clutter.
- It repairs words split across lines.
- It says "e.g.", "Fig. 3", "$3.50", "1990s", and "H₂O" the way a person would.

It then reads everything aloud in a natural voice and highlights each sentence as it
goes. Click any sentence to jump there, change the speed, and pick up where you left
off. You can also save chapters as MP3s for your phone.

There's an optional **study helper** powered by Claude. It can explain a confusing
passage, answer questions about what you're reading, summarize a section, quiz you,
read tables and equations in plain words, and describe the figures on a page.

## Privacy first

* **Everything runs on your computer.** Your books, reading progress, settings and
  audio stay in a private folder in your user account. Nothing is uploaded. There is
  no tracking, no analytics, and no hidden "phone home": Athena even switches off the
  usage reporting built into the voice engine it uses. The reader only listens on
  `127.0.0.1`, so other devices can't reach it.
* **Online features are off until you turn them on.** A ☁ badge shows when one is on.
  * *Online voices* (Microsoft Edge voices) send the text being read to Microsoft.
  * The *study helper* sends the text you ask about to Anthropic (Claude). It needs
    your own API key, which is stored only in your local settings file and never shown
    in the browser.
* **Easy to erase.** Settings → Privacy → *Erase all my data* (or `athena wipe`)
  deletes your books, progress, settings, API key, and the reader's browser storage.

## Install

### Windows: the installer (easiest)

1. Download `Athena-Setup-<version>.exe`. GitHub builds it on every release: see
   **Releases**, or the **desktop** workflow's artifacts under **Actions**.
2. Double-click it. Windows may show a blue **"Windows protected your PC"** box, because
   the installer isn't code-signed. Click **More info → Run anyway**.
3. Click **Next** through the installer. You don't need an administrator password,
   and both natural voices are already included.
4. Open **Athena** from the desktop icon or the Start menu.

To uninstall, use **Settings → Apps → Athena**. It asks whether to delete your books
and settings too.

### From the source folder (Windows, Mac, Linux)

You need **Python 3.10 or newer**: <https://www.python.org/downloads/>. On Windows,
tick **"Add python.exe to PATH"** in the installer.

1. Download this folder: **Code → Download ZIP** on GitHub, then unzip it. Or use `git clone`.
2. Start it:
   * **Windows:** double-click `start-windows.bat`.
   * **Mac:** double-click `start-mac.command`. The first time, right-click it →
     *Open* → *Open*, because macOS asks about downloaded scripts.
   * **Linux:** run `./start-linux.sh`.
3. The first launch takes a few minutes. It sets up a private Python environment in
   the folder (`.venv`), downloads the natural offline voices (about 180 MB), and adds
   an **Athena** icon to your desktop. Later, just use that icon.

## Using the reader

| Do this | How |
|---|---|
| Add a book or handout | **Add material**, drag a file onto the window, or **Paste text** |
| Play / pause | ▶ button or `Space` |
| Previous / next sentence | ⏮ ⏭ or `←` `→` |
| Previous / next section | `Shift+←` `Shift+→`, or click the table of contents |
| Start from a sentence | Click it |
| Speed | The speed slider or `+` / `-`, from 0.5× to 3×. The voice stays natural. |
| Bookmark | 🔖 or `B` |
| All shortcuts | `?` |
| Save as audio (MP3) | **Save as audio**: pick chapters (it shows how long that will take) and get MP3 files with a playlist |

**What gets read** can be changed in Settings → Reading. You can choose headings,
figure captions, footnotes (off by default), tables (row by row, announced only, or
skipped), equations, and code examples. Citations like "(Smith et al., 2019)" and "[12]"
are skipped by default, and web links are shortened to "a link to example dot com".

### Voices

| Voice | Sound | Internet? |
|---|---|---|
| **Kokoro**: the default when installed | The most natural | No |
| **Piper**: natural, and very fast on slower laptops | Natural | No |
| **System voice**: the voices built into Windows, macOS or Linux | Robotic | No |
| **Browser voice**: your web browser's voices | Varies | Depends on the browser |
| **Microsoft online voices**: Edge neural voices | Excellent | **Yes**. Off until you allow online voices |

If a laptop is too slow for Kokoro at the chosen speed, Athena notices and offers to
switch to Piper. You can download more Piper voices (US and UK English, Spanish) in
Settings → Voice.

### Supported files

PDF (including two-column layouts, tables, and scanned pages through OCR), EPUB, Word
(`.docx`), PowerPoint (`.pptx`, with speaker notes), web pages (`.html`), text and
Markdown (`.txt`, `.md`), and photos or scans of pages (`.png`, `.jpg`, `.tif`, and
others).

Scanned pages and photos need the free **Tesseract OCR** program:
* **Windows:** the installer from <https://github.com/UB-Mannheim/tesseract/wiki>
* **Mac:** `brew install tesseract`
* **Linux:** `sudo apt install tesseract-ocr`

DRM-protected e-books, old `.doc`/`.ppt` files, and password-protected files can't be
opened. Athena explains what to do instead, for example: "Use Save As to make a .docx copy".

### The study helper (optional)

1. Get an API key at <https://console.anthropic.com/>. Usage is billed by Anthropic.
2. In Athena, open Settings → Privacy & AI, turn on **Study helper**, and paste the key.
3. Open the **Study helper** panel while reading to explain, ask, summarize, take a
   quiz, or describe the figures on a PDF page. Every answer can be read aloud.

Only the current section, or the passage or page you ask about, is sent. The default
model is `claude-opus-5-5`, and you can change it in Settings.

## Command line

The source-folder setup also installs an `athena` command inside `.venv`. Run
`.venv/bin/athena ...`, or `.venv\Scripts\athena ...` on Windows.

```
athena app                   open the reader in its own window
athena serve                 start the reader in your web browser instead
athena import notes.pdf      add files to the library
athena list                  show the library
athena toc "biology"         table of contents of a book (id or part of the title)
athena read "biology" --section s3 --rate 1.25    read aloud in the terminal
athena export "biology" --format mp3 --out ~/Music/Biology
athena voices catalog        downloadable natural voices
athena voices download en_US-amy-medium
athena doctor                check what's installed and working
athena self-test             a full check of voices, files and the reader, with timings
athena wipe                  erase all Athena data
```

Data is stored in `%LOCALAPPDATA%\Athena` on Windows, in
`~/Library/Application Support/Athena` on macOS, and in `~/.local/share/athena` on Linux.
Set `ATHENA_HOME` to use another folder.

## Troubleshooting

* **Run `athena doctor`, or `athena self-test` for the full check, first.** It lists
  every voice engine, OCR and the AI setup, and says what's missing.
* **"No natural voice downloaded yet"**: open Settings → Voice and download *Kokoro*
  or *Lessac*, or run `athena voices download kokoro-v1.0`.
* **A page was skipped or reads oddly**: some PDFs are scans or have unusual layouts.
  Install Tesseract for scans, and read the notice shown after importing.
* **"Loading the voice…" appears often**: the laptop is slow for that voice at that
  speed. Accept the offer to switch to Piper, or lower the speed.
* **The browser voice says nothing**: some browsers have no voices installed. Use
  Kokoro, Piper or the system voice instead.

## For developers

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -m "not e2e"          # the main suite
pytest tests/e2e -m e2e      # the real reader in Chromium with real voices (see tests/e2e/harness.py)
```

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how the pieces fit together.
[docs/API.md](docs/API.md) documents the local JSON API.
[packaging/README.md](packaging/README.md) covers building the desktop app and installer.
[docs/HANDOFF-WINDOWS.md](docs/HANDOFF-WINDOWS.md) is the checklist for testing the voices
on a real Windows laptop.

Third-party components keep their own licenses. PyMuPDF (PDF reading) is AGPL and
Piper is GPL. Both are fine for personal use.
