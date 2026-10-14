# Packaging Athena as a desktop app

This folder turns Athena into an app that a non-technical person can install
with a double-click. They don't need Python, there's no console window, and two
natural offline voices are built in: Kokoro (the most natural) and Piper Lessac.
It works fully offline from the first launch.

| Platform | What you get | How it opens |
|---|---|---|
| Windows 10/11 | `dist/Athena-Setup-<version>.exe`: a per-user installer (no admin password) | "Athena" icon on the Desktop and in the Start menu; the reader opens in its own window (WebView2) |
| macOS 11+ | `dist/Athena-<version>.dmg` containing `Athena.app` | Drag to Applications; opens in its own window (WebKit) |
| Linux | `dist/Athena-<version>-linux-<arch>.tar.gz` (mainly used to test the build) | `Athena/Athena` |

Closing the window quits Athena. Starting it again while it's already running
opens a window onto the running copy instead of starting a second one.

## Building

You need Python 3.10 or newer (3.12 recommended). On Windows you also need
[Inno Setup 6.3 or newer](https://jrsoftware.org/isdl.php)
(`winget install JRSoftware.InnoSetup`).

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
```

```bash
# macOS
bash packaging/build_macos.sh
```

Both scripts make a private build environment in `packaging/build/venv` and
then run `packaging/build.py`. If you already have an environment with
`pip install -e ".[build]"`, you can run that script directly:

```bash
python packaging/build.py                  # bundles Piper Lessac and the Kokoro voice pack
python packaging/build.py --no-kokoro      # Piper only (about 100 MB smaller installer)
python packaging/build.py --voice-source ~/voices/piper   # copy the Piper voice instead of downloading it
```

`build.py` runs these steps:

1. **Voices.** Downloads Piper `en_US-lessac-medium` (63 MB) from Hugging Face
   into `packaging/build/voices/piper/`, using
   `destiny_bot.tts.catalog.download_piper_voice`. It also downloads the Kokoro
   int8 voice pack (`kokoro-v1.0.int8.onnx`, 92 MB, and `voices-v1.0.bin`, 28 MB)
   from the kokoro-onnx GitHub release into `packaging/build/voices/kokoro/`,
   using `download_kokoro_model(variant="int8")`, which checks the SHA-256 of each
   file. Use `--no-kokoro` or `ATHENA_BUNDLE_KOKORO=0` to leave Kokoro out.
2. **Icons.** Runs `make_icons.py`, which draws the book, sound waves and laurel
   icon with Pillow, matching `src/destiny_bot/web/icon.svg`. It writes
   `icons/icon.ico`, `icon.icns` and `icon-512.png`, which are committed. It also
   writes the installer's wizard images into `build/wizard/`. Sizes of 32 px and
   below use a simpler, bolder drawing so the icon stays readable in the taskbar.
3. **PyInstaller.** Builds from `destiny.spec`: one folder, no console, named
   "Athena". The entry point is `destiny_app.py`. The spec collects the reader's
   web files, the Piper/espeak-ng data, the onnxruntime libraries, PyMuPDF,
   the python-docx/python-pptx templates, certifi, edge-tts, uvicorn's
   dynamically loaded modules, pywebview with this platform's backend
   (pythonnet on Windows), kokoro-onnx, espeakng-loader (its espeak-ng library
   and data) and phonemizer, plus the voices from step 1 (in `_internal/voices`,
   which the voice engines search read-only through `DESTINY_EXTRA_VOICE_DIRS`).
4. **Smoke test.** Runs the built app with `--self-test-quick` in a throwaway
   data folder. Both bundled voices must speak, or the build stops. The report
   is copied to `dist/self-test-<platform>.json`. For each offline engine it
   records the first-sentence latency cold (`latency_cold_s`, which includes
   loading the voice) and warm (`latency_warm_s`), and the real-time factor
   (`rtf_cold`, `rtf_warm`; below 1 means faster than speech).
5. **Package.** On Windows, Inno Setup compiles `windows/DestinyBot.iss`. On
   macOS, `hdiutil` makes the `.dmg`. On Linux, it makes a `.tar.gz`.

## How the app works

- `destiny_app.py` is the frozen entry point. With no arguments it calls
  `destiny_bot.desktop.run_app()`. With `--self-test` or `--self-test-quick` it
  runs `destiny_bot.selftest.run_self_test()` and writes
  `<data>/logs/self-test.json` and `self-test.txt`.
- `run_app()` sends stdout and stderr to `<data>/logs/app.log`, which is rotated
  and keeps 3 old copies. Crashes inside native libraries go to
  `<data>/logs/crash.log`. It points `DESTINY_EXTRA_VOICE_DIRS` at the bundled
  voices, starts the server on `127.0.0.1` in a background thread, and shows it
  in a pywebview window. Page storage is kept in `<data>/webview`, so the
  reader's preferences survive restarts. Downloads (exported MP3s) get a normal
  Save dialog.
- If something goes wrong before the window opens, the user sees a plain
  message box that says where the log file is.
- **Fallbacks.** If pywebview or WebView2 is unavailable, or the native window
  fails, Athena opens in Microsoft Edge or Google Chrome "app mode": a window
  with no tabs or address bar, using its own profile in
  `<data>/browser-profile`. Athena keeps running until that window is closed.
  As a last resort it opens the default browser.
- **Data folder.** This is `config.data_dir()`: `%LOCALAPPDATA%\Athena` on
  Windows (after the integrator's rename sweep) and
  `~/Library/Application Support/...` on macOS. The installer never touches it,
  except when the user asks the uninstaller to delete it.

### WebView2 (Windows)

Windows 11, and Windows 10 with current updates, already include the Microsoft
Edge WebView2 runtime that draws Athena's window. The installer checks for it
with the registry keys Microsoft documents for the Evergreen runtime:
`HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}`
and the `HKCU` equivalent, value `pv`. If the runtime is missing, the installer
explains this and offers to download Microsoft's 2 MB Evergreen bootstrapper and
run it silently. That installs WebView2 per user, with no admin rights needed.
If the user says no, or there's no internet, Athena still works through the
Edge app-mode fallback, because Edge is part of Windows. `run_app()` does the
same registry check at startup. Without it, pywebview would fall back to the
old Internet Explorer engine, which can't run the reader.

## The Windows installer (`windows/DestinyBot.iss`)

- Installs per user (`PrivilegesRequired=lowest`) to
  `%LOCALAPPDATA%\Programs\Athena`, so no admin password is needed.
- Adds a Start menu entry and a Desktop icon. The Desktop icon is ticked by
  default. Both shortcuts carry the AppUserModelID `Athena.Reader`, the same one
  the app sets, so pinned taskbar icons group correctly.
- Ends with "Open Athena now". Registers an uninstaller under
  Settings > Apps > Installed apps.
- Updates replace `_internal\` completely. If Athena is running, Setup asks the
  user to close it first (`AppMutex=AthenaReaderAppMutex`, which `run_app`
  creates).
- When uninstalling, it asks "Also delete your books, reading progress and
  settings?". The default answer is No. It deletes `%LOCALAPPDATA%\Athena` only
  if the user says yes, and silent uninstalls never delete it.

## Code signing (recommended before giving it to people)

The builds aren't signed, so:

- **Windows:** SmartScreen may say "Windows protected your PC". The user has to
  click *More info*, then *Run anyway*. To avoid that, sign `Athena.exe` and the
  installer with an OV/EV code-signing certificate or Azure Trusted Signing. Use
  `signtool sign /fd sha256 /tr http://timestamp.digicert.com /td sha256 ...`,
  or set Inno Setup's `SignTool`. Signing also helps with antivirus false
  positives, which are common with PyInstaller apps.
- **macOS:** Gatekeeper blocks unsigned apps that were downloaded. The user has
  to right-click `Athena.app`, choose *Open*, then *Open*. To avoid that, sign
  with a Developer ID (`codesign --deep --options runtime`), notarize with
  `xcrun notarytool`, and staple the ticket.

## Continuous integration

- `.github/workflows/tests.yml` runs pytest on Ubuntu and Windows with Python
  3.12, skipping the `e2e` tests. It uses a real Piper voice, and on Windows the
  real SAPI system-voice code paths. It then runs `destiny self-test` and
  uploads the report.
- `.github/workflows/desktop.yml` runs on changes to `packaging/` or `src/`, on
  `v*` tags, and on demand. The Windows job builds the installer and runs the
  smoke test. It then installs the app silently, checks the Desktop and Start
  menu shortcuts and the Apps & features entry, and runs the installed app's
  full `--self-test`. Next, `check_window.py` starts `Athena.exe` the way a
  double-click does. It checks that the "Athena" window and WebView2 come up
  and takes a screenshot. It starts the app a second time and checks that this
  opens a second window onto the same server. It then closes both windows and
  checks that the app quits and removes its lock file. Finally the job
  uninstalls the app. It uploads `Athena-Setup-windows` (the
  installer) and `athena-windows-self-test` (the JSON reports). On a `v*` tag it
  attaches both to the GitHub Release. The macOS job runs only on demand, to
  save minutes, and uploads the `.dmg`.

## Diagnosing a problem on someone's computer

Ask them to send the files in the `logs` folder inside the data folder:
`app.log`, `crash.log`, and `self-test.json` if they ran a self-test. To run a
self-test on Windows:

```powershell
& "$env:LOCALAPPDATA\Programs\Athena\Athena.exe" --self-test
```

On macOS, run `/Applications/Athena.app/Contents/MacOS/Athena --self-test`.
