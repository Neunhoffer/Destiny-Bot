# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the Athena desktop app (one folder, no console window).

Build with ``python packaging/build.py`` (which also fetches the bundled voice and
runs a smoke test), or directly:

    python -m PyInstaller --noconfirm --clean packaging/destiny.spec

Output: dist/Athena/Athena.exe (Windows), dist/Athena.app (macOS), dist/Athena/Athena (Linux).

Environment:
    ATHENA_VOICES_DIR      voices to bundle (default packaging/build/voices; holds piper/ and kokoro/)
    ATHENA_BUNDLE_KOKORO   "1" to include voices/kokoro (the ~120 MB Kokoro model) when present
"""

import os
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

HERE = Path(SPECPATH).resolve()
ROOT = HERE.parent
SRC = ROOT / "src"
APP_NAME = "Athena"
BUNDLE_ID = "com.athena.reader"
VERSION = re.search(r'__version__\s*=\s*"([^"]+)"', (SRC / "destiny_bot" / "__init__.py").read_text()).group(1)
VOICES = Path(os.environ.get("ATHENA_VOICES_DIR") or HERE / "build" / "voices")
BUNDLE_KOKORO = os.environ.get("ATHENA_BUNDLE_KOKORO", "").strip().lower() in ("1", "true", "yes")
IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

datas = []
binaries = []
hiddenimports = []


def add_all(package, **kwargs):
    """collect_all() for an optional package; skipped when it isn't installed."""
    try:
        __import__(package)
    except Exception as exc:  # optional extras may be missing on this build machine
        print(f"[athena] not bundling {package}: {exc}")
        return
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package, **kwargs)
    datas.extend(pkg_datas)
    binaries.extend(pkg_binaries)
    hiddenimports.extend(pkg_hidden)


def installed(package):
    try:
        __import__(package)
        return True
    except Exception:
        return False


# Athena itself: every module (some are imported lazily), the reader's web files, the icons.
hiddenimports += collect_submodules("destiny_bot")
datas += [(str(SRC / "destiny_bot" / "web"), "destiny_bot/web"), (str(HERE / "icons"), "icons")]

# The local web server. uvicorn loads its loop/protocol/logging classes by name.
hiddenimports += collect_submodules("uvicorn")
hiddenimports += ["python_multipart", "multipart"]

# File readers: PyMuPDF (PDF), lxml + python-docx/python-pptx templates, ebooklib.
binaries += collect_dynamic_libs("pymupdf")
hiddenimports += collect_submodules("pymupdf")
datas += collect_data_files("docx") + collect_data_files("pptx")
hiddenimports += ["lxml.etree", "lxml._elementpath", "lxml.html", "bs4", "ebooklib", "ebooklib.epub"]

# Voices. Piper: espeak-ng data and the espeak bridge (the Hebrew model and training code aren't needed).
if installed("piper"):
    datas += collect_data_files("piper", excludes=["hebrew/**", "train/**", "img/**", "templates/**"])
    binaries += collect_dynamic_libs("piper")
    hiddenimports += ["piper", "piper.voice", "piper.espeakbridge", "piper.tashkeel", "piper.phonemize_espeak"]
if installed("onnxruntime"):
    binaries += collect_dynamic_libs("onnxruntime")
    hiddenimports += ["onnxruntime", "onnxruntime.capi._pybind_state"]
for optional in ("kokoro_onnx", "espeakng_loader", "phonemizer"):
    add_all(optional)
if installed("edge_tts"):
    hiddenimports += collect_submodules("edge_tts")
    datas += collect_data_files("certifi")
if IS_WIN:
    hiddenimports += ["pyttsx3", "pyttsx3.drivers", "pyttsx3.drivers.sapi5", "comtypes.client", "pythoncom", "pywintypes"]
elif IS_MAC:
    hiddenimports += ["pyttsx3.drivers.nsss"] if installed("pyttsx3") else []

# OCR (needs the Tesseract program on the computer) and the AI helpers.
hiddenimports += ["pytesseract", "PIL.Image"]
if installed("anthropic"):
    hiddenimports += ["anthropic"]

# The native window: pywebview with only this platform's backend.
if installed("webview"):
    keep = {"winforms", "edgechromium", "win32", "mshtml"} if IS_WIN else {"cocoa"} if IS_MAC else {"gtk", "qt"}
    skip = {"winforms", "edgechromium", "win32", "mshtml", "cocoa", "gtk", "qt", "cef", "android"} - keep
    add_all("webview", filter_submodules=lambda name: not any(f"platforms.{p}" in name for p in skip))
    if IS_WIN:
        add_all("clr_loader")
        add_all("pythonnet")
        hiddenimports += ["clr"]

# Voices shipped inside the app (searched read-only through DESTINY_EXTRA_VOICE_DIRS).
for kind in ("piper", "kokoro"):
    folder = VOICES / kind
    if kind == "kokoro" and not BUNDLE_KOKORO:
        continue
    if folder.is_dir() and any(folder.glob("*.onnx")):
        for path in sorted(folder.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                datas.append((str(path), f"voices/{kind}"))
                print(f"[athena] bundling voice file {kind}/{path.name} ({path.stat().st_size / 1e6:.1f} MB)")

excludes = [
    "tkinter",
    "_tkinter",
    "test",
    "tests",
    "pytest",
    "_pytest",
    "playwright",
    "IPython",
    "matplotlib",
    "scipy",
    "pandas",
    "torch",
    "tensorflow",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "qtpy",
    "cefpython3",
    "kivy",
    "piper.train",
    "piper.http_server",
    "flask",
    "onnx",
    "g2pw",
    "transformers",
]
if not IS_WIN:
    excludes += ["pythonnet", "clr", "clr_loader", "comtypes", "win32com", "pythoncom"]
# pywebview backends for other platforms (they are imported by name only when chosen).
_backends = {"win32": ["winforms", "edgechromium", "win32", "mshtml"], "darwin": ["cocoa"], "linux": ["gtk", "qt"]}
_mine = _backends["win32" if IS_WIN else "darwin" if IS_MAC else "linux"]
excludes += [f"webview.platforms.{name}" for names in _backends.values() for name in names if name not in _mine]
excludes += ["webview.platforms.android", "webview.platforms.cef"]

a = Analysis(
    [str(HERE / "destiny_app.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

version_file = None
if IS_WIN:
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo,
        StringFileInfo,
        StringStruct,
        StringTable,
        VarFileInfo,
        VarStruct,
        VSVersionInfo,
    )

    parts = tuple(int(p) for p in (re.findall(r"\d+", VERSION) + ["0", "0", "0", "0"])[:4])
    info = VSVersionInfo(
        ffi=FixedFileInfo(filevers=parts, prodvers=parts),
        kids=[
            StringFileInfo(
                [
                    StringTable(
                        "040904B0",
                        [
                            StringStruct("CompanyName", APP_NAME),
                            StringStruct("FileDescription", f"{APP_NAME} - reads your textbooks aloud"),
                            StringStruct("FileVersion", VERSION),
                            StringStruct("InternalName", APP_NAME),
                            StringStruct("OriginalFilename", f"{APP_NAME}.exe"),
                            StringStruct("ProductName", APP_NAME),
                            StringStruct("ProductVersion", VERSION),
                        ],
                    )
                ]
            ),
            VarFileInfo([VarStruct("Translation", [1033, 1200])]),
        ],
    )
    version_file = str(Path(workpath) / "version_info.txt")
    Path(version_file).parent.mkdir(parents=True, exist_ok=True)
    Path(version_file).write_text(str(info), encoding="utf-8")

icon = str(HERE / "icons" / ("icon.ico" if IS_WIN else "icon.icns" if IS_MAC else "icon-512.png"))

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
    version=version_file,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=APP_NAME)

if IS_MAC:
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=icon,
        bundle_identifier=BUNDLE_ID,
        version=VERSION,
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "LSApplicationCategoryType": "public.app-category.education",
            "LSMinimumSystemVersion": "11.0",
            "NSHumanReadableCopyright": "Athena reads your textbooks aloud. Private and offline.",
        },
    )
