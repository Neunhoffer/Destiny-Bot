# Destiny Bot 📖🔊

**A private study reader that reads your textbooks and class materials out loud.**

Open a PDF textbook, an e-book, Word notes, lecture slides, or a photo of a handout.
Destiny Bot cleans the text up for listening: it skips page numbers and running headers,
repairs words split across lines, and turns "e.g." and "Fig. 3" into words. Then it reads
everything aloud in a natural voice and highlights each sentence as it goes. Click any
sentence to jump there, change the speed, and pick up where you left off.
You can also save chapters as MP3s for your phone.

There's an optional **study helper** powered by Claude. It can explain a confusing
passage, answer questions about what you're reading, summarize a section, quiz you,
read tables and equations in plain words, and describe the figures on a page.

## Privacy first

* **Everything runs on your computer.** Your books, reading progress, settings and
  audio are kept in one local folder. Nothing is uploaded, and there is no tracking or
  analytics. The reader only listens on `127.0.0.1`, so other devices can't reach it.
* **Online features are off until you turn them on**, and the app shows a ☁ badge when one is on:
  * *Online voices* (Microsoft Edge voices) send the text being read to Microsoft.
  * The *study helper* sends the text you ask about to Anthropic (Claude). It needs
    your own API key, which is stored only in your local settings file and never shown
    in the browser.
* **Easy to erase.** Settings → Privacy → *Erase all my data* (or `destiny wipe`)
  deletes your books, progress, settings, and API key. Deleting the Destiny Bot
  folder removes everything.

## Quick start

You need **Python 3.10 or newer**: <https://www.python.org/downloads/>. On Windows,
tick **"Add python.exe to PATH"** in the installer.

1. Download this folder: **Code → Download ZIP** on GitHub, then unzip it. Or use `git clone`.
2. Start it:
   * **Windows:** double-click `start-windows.bat`.
   * **Mac:** double-click `start-mac.command`. The first time, right-click it →
     *Open* → *Open*, because macOS asks about downloaded scripts.
   * **Linux:** run `./start-linux.sh`.
3. The first launch takes a few minutes. It sets up a private Python environment
   inside the folder and downloads a natural-sounding offline voice (about 60 MB).
   Later launches take a few seconds.
4. Your browser opens the reader. Click **Add material**, choose a file, and press ▶.

To stop Destiny Bot, close the black terminal window, or press `Ctrl+C` in it.

The launchers keep everything inside the Destiny Bot folder: the app in `.venv`, and
your books and settings in `data`. To uninstall completely, delete the folder.

## Using the reader

| Do this | How |
|---|---|
| Add a book or handout | **Add material**, drag a file onto the window, or **Paste text** |
| Play / pause | ▶ button or `Space` |
| Previous / next sentence | ⏮ ⏭ or `←` `→` |
| Previous / next section | `Shift+←` `Shift+→`, or click the table of contents |
| Start from a sentence | Click it |
| Speed | The speed slider or `+` / `-`. Pitch stays natural. |
| Bookmark | 🔖 or `B` |
| All shortcuts | `?` |
| Save as audio (MP3) | **Save as audio**: pick chapters and get MP3 files with a playlist |

**What gets read** can be changed in Settings → Reading. You can choose headings,
figure captions, footnotes (off by default), tables (row by row, announced only, or
skipped), equations, and code examples. Citations like "(Smith et al., 2019)" and "[12]"
are skipped by default, and web links are shortened to "a link to example.com".

### Voices

| Voice | Sound | Internet? |
|---|---|---|
| **Piper**: natural neural voices (recommended) | Very natural | No. Runs offline after a one-time download |
| **System voice**: Windows, macOS, or Linux built-in voices | OK | No |
| **Browser voice**: your web browser's voices | Varies | Depends on the browser. Some browsers use online voices |
| **Microsoft online voices**: Edge neural voices | Excellent | **Yes**. Off until you allow online voices |

You can download more Piper voices (US and UK English, Spanish) in Settings → Voice.

### Supported files

PDF (including two-column layouts, and scanned pages with OCR), EPUB, Word (`.docx`),
PowerPoint (`.pptx`, with speaker notes), web pages (`.html`), text and Markdown
(`.txt`, `.md`), and images (`.png`, `.jpg`, `.tif`, and others) through OCR.

Scanned pages and photos need the free **Tesseract OCR** program:
* **Windows:** the installer from <https://github.com/UB-Mannheim/tesseract/wiki>
* **Mac:** `brew install tesseract`
* **Linux:** `sudo apt install tesseract-ocr`

DRM-protected e-books can't be opened.

### The study helper (optional)

1. Get an API key at <https://console.anthropic.com/>. Usage is billed by Anthropic.
2. In Destiny Bot, open Settings → Privacy & AI, turn on **Study helper**, and paste the key.
3. Open the **Study helper** panel while reading to explain, ask, summarize, take a
   quiz, or describe the figures on a PDF page. Every answer can be read aloud.

Only the current section, or the passage or page you ask about, is sent. The default
model is `claude-opus-5-5`, and you can change it in Settings.

## Command line

The launchers install a `destiny` command inside `.venv`. Run
`.venv/bin/destiny ...`, or `.venv\Scripts\destiny ...` on Windows.

```
destiny serve                 start the reader (default)
destiny import notes.pdf      add files to the library
destiny list                  show the library
destiny toc "biology"         table of contents of a book (id or part of the title)
destiny read "biology" --section s3 --rate 1.25    read aloud in the terminal
destiny export "biology" --format mp3 --out ~/Music/Biology
destiny voices catalog        downloadable natural voices
destiny voices download en_US-amy-medium
destiny doctor                check what's installed and working
destiny wipe                  erase all Destiny Bot data
```

Set `DESTINY_HOME` to choose where data is stored. Otherwise it goes to
`%LOCALAPPDATA%\DestinyBot` on Windows, `~/Library/Application Support/DestinyBot` on
macOS, and `~/.local/share/destiny-bot` on Linux. The launchers set it to the `data`
folder.

## Troubleshooting

* **Run `destiny doctor` first.** It lists every voice engine, OCR, and the AI
  setup, and tells you what's missing.
* **"No natural voice downloaded yet"**: Settings → Voice → download *Lessac*, or run
  `destiny voices download en_US-lessac-medium`.
* **A page was skipped or reads oddly**: some PDFs are scans or have unusual layouts.
  Install Tesseract for scans, and check the notice shown after importing.
* **The browser voice says nothing**: some browsers have no voices installed. Use
  Piper or the system voice instead.

## For developers

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how the pieces fit together.
[docs/API.md](docs/API.md) documents the local JSON API.

Third-party components keep their own licenses. PyMuPDF (PDF reading) is AGPL and
Piper is GPL. Both are fine for personal use.
