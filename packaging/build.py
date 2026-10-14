"""Build the Athena desktop app and its installer.

Steps:
    1. put the bundled voices in packaging/build/voices: Piper en_US-lessac-medium in
       piper/ and the Kokoro int8 voice pack in kokoro/ (leave Kokoro out with
       --no-kokoro or ATHENA_BUNDLE_KOKORO=0)
    2. draw the icons (packaging/make_icons.py)
    3. run PyInstaller (packaging/destiny.spec) -> dist/Athena (dist/Athena.app on macOS)
    4. smoke test: run the built app with --self-test-quick; it must pass
    5. package it: Windows -> dist/Athena-Setup-<version>.exe (Inno Setup),
       macOS -> dist/Athena-<version>.dmg (hdiutil), Linux -> dist/Athena-<version>-linux-<arch>.tar.gz

Run from the repository root with the build extras installed:
    python -m pip install -e ".[build]"
    python packaging/build.py [--no-kokoro] [--voice-source DIR] [--skip-installer] [--skip-smoke-test]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = HERE / "build"
VOICES = BUILD / "voices"
DIST = ROOT / "dist"
APP_NAME = "Athena"
PIPER_VOICE = "en_US-lessac-medium"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))


def version() -> str:
    text = (ROOT / "src" / "destiny_bot" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise SystemExit("Couldn't find __version__ in src/destiny_bot/__init__.py")
    return match.group(1)


def step(title: str) -> None:
    print(f"\n=== {title}", flush=True)


_last_progress = [-1]


def _progress(fraction: float, message: str) -> None:
    if sys.stdout.isatty():
        print(f"  {fraction * 100:5.1f}%  {message}", end="\r", flush=True)
        return
    step = int(fraction * 10)  # CI logs: one line per 10%
    if step != _last_progress[0]:
        _last_progress[0] = step
        print(f"  {fraction * 100:5.1f}%  {message}", flush=True)


def fetch_voices(*, kokoro: bool, source: Path | None) -> None:
    from destiny_bot.tts import catalog

    piper_dir = VOICES / "piper"
    piper_dir.mkdir(parents=True, exist_ok=True)
    model = piper_dir / f"{PIPER_VOICE}.onnx"
    if source is not None:
        for name in (f"{PIPER_VOICE}.onnx", f"{PIPER_VOICE}.onnx.json"):
            candidates = [source / name, source / "piper" / name]
            found = next((c for c in candidates if c.is_file()), None)
            if found is None:
                raise SystemExit(f"{name} isn't in {source}")
            shutil.copyfile(found, piper_dir / name)
    elif model in catalog.installed_piper_voices(piper_dir):
        print(f"  {PIPER_VOICE} is already here")
    else:
        catalog.download_piper_voice(PIPER_VOICE, voices_dir=piper_dir, progress=_progress)
        print()
    for path in sorted(piper_dir.iterdir()):
        print(f"  piper/{path.name}: {path.stat().st_size / 1e6:.1f} MB")

    if kokoro:
        kokoro_dir = VOICES / "kokoro"
        kokoro_dir.mkdir(parents=True, exist_ok=True)
        model = catalog.download_kokoro_model(voices_dir=kokoro_dir, variant="int8", progress=_progress)
        print()
        # The files must sit directly in voices/kokoro/, wherever the downloader put them.
        table = model.with_name(catalog.KOKORO_FILES["voices"].name)
        for path in (model, table):
            if path.parent != kokoro_dir:
                shutil.move(str(path), kokoro_dir / path.name)
        (kokoro_dir / catalog.KOKORO_FILES["fp32"].name).unlink(missing_ok=True)  # only int8 is bundled
        if catalog.find_kokoro_model(kokoro_dir, variant="int8") is None:
            raise SystemExit(f"The Kokoro voice pack isn't complete in {kokoro_dir}")
        for path in sorted(p for p in kokoro_dir.iterdir() if p.is_file()):
            print(f"  kokoro/{path.name}: {path.stat().st_size / 1e6:.1f} MB")
    else:
        shutil.rmtree(VOICES / "kokoro", ignore_errors=True)


def make_icons() -> None:
    import make_icons

    for path in make_icons.write_icons(HERE / "icons"):
        print(f"  {path.relative_to(ROOT)}")
    if sys.platform == "win32":
        for path in make_icons.write_wizard_images(BUILD / "wizard"):
            print(f"  {path.relative_to(ROOT)}")


def run_pyinstaller(*, kokoro: bool) -> Path:
    env = {**os.environ, "ATHENA_VOICES_DIR": str(VOICES), "ATHENA_BUNDLE_KOKORO": "1" if kokoro else "0"}
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--log-level",
        "WARN",
        "--distpath",
        str(DIST),
        "--workpath",
        str(BUILD / "pyinstaller"),
        str(HERE / "destiny.spec"),
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    if sys.platform == "darwin":
        return DIST / f"{APP_NAME}.app"
    return DIST / APP_NAME


def app_executable(app: Path) -> Path:
    if sys.platform == "darwin":
        return app / "Contents" / "MacOS" / APP_NAME
    return app / (f"{APP_NAME}.exe" if sys.platform == "win32" else APP_NAME)


def folder_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def smoke_test(app: Path) -> Path:
    """Run the built app's quick self-test in a throwaway data folder; exits on failure."""
    exe = app_executable(app)
    home = Path(tempfile.mkdtemp(prefix="athena-smoke-"))
    env = {**os.environ, "DESTINY_HOME": str(home)}
    env.pop("DESTINY_EXTRA_VOICE_DIRS", None)  # only the voices inside the app may count
    started = time.monotonic()
    proc = subprocess.run([str(exe), "--self-test-quick"], env=env, timeout=900, check=False)
    elapsed = time.monotonic() - started
    report_txt = home / "logs" / "self-test.txt"
    report_json = home / "logs" / "self-test.json"
    for log_name in ("self-test.txt", "self-test.log", "crash.log"):
        log_path = home / "logs" / log_name
        if log_path.is_file() and log_path.stat().st_size:
            print(f"--- {log_name}")
            print(log_path.read_text(encoding="utf-8", errors="replace"))
            if log_name == "self-test.txt":
                break
    print(f"Self-test exit code {proc.returncode} after {elapsed:.1f} s")
    tag = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    saved = DIST / f"self-test-{tag}.json"
    if report_json.is_file():
        shutil.copyfile(report_json, saved)
    if proc.returncode != 0 or not report_json.is_file():
        raise SystemExit("The built app's self-test failed (see the report above).")
    report = json.loads(report_json.read_text(encoding="utf-8"))
    print(f"Smoke test passed: {report['summary']}")
    if not report_txt.is_file():
        print("(no text report was written)")
    shutil.rmtree(home, ignore_errors=True)
    return saved


def find_iscc() -> Path | None:
    found = shutil.which("iscc") or shutil.which("ISCC.exe")
    if found:
        return Path(found)
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        if not base:
            continue
        for rel in (r"Inno Setup 6\ISCC.exe", r"Programs\Inno Setup 6\ISCC.exe"):
            candidate = Path(base) / rel
            if candidate.is_file():
                return candidate
    return None


def windows_installer(app: Path, ver: str) -> Path:
    iscc = find_iscc()
    if iscc is None:
        raise SystemExit(
            "Inno Setup 6 isn't installed. Get it from https://jrsoftware.org/isdl.php "
            "(or run: winget install JRSoftware.InnoSetup), then run this again."
        )
    numeric = ".".join((re.findall(r"\d+", ver) + ["0", "0", "0"])[:3])
    cmd = [
        str(iscc),
        "/Qp",
        f"/DAppVersion={ver}",
        f"/DAppNumericVersion={numeric}",
        f"/DSourceDir={app}",
        f"/DOutputDir={DIST}",
        f"/DIconFile={HERE / 'icons' / 'icon.ico'}",
    ]
    wizard = BUILD / "wizard"
    if (wizard / "small-100.bmp").is_file():
        small = ",".join(str(p) for p in sorted(wizard.glob("small-*.bmp")))
        large = ",".join(str(p) for p in sorted(wizard.glob("large-*.bmp")))
        cmd += [f"/DWizardSmall={small}", f"/DWizardLarge={large}"]
    cmd.append(str(HERE / "windows" / "DestinyBot.iss"))
    subprocess.run(cmd, check=True)
    return DIST / f"{APP_NAME}-Setup-{ver}.exe"


def macos_dmg(app: Path, ver: str) -> Path:
    dmg = DIST / f"{APP_NAME}-{ver}.dmg"
    dmg.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="athena-dmg-") as tmp:
        staging = Path(tmp) / APP_NAME
        staging.mkdir()
        subprocess.run(["ditto", str(app), str(staging / app.name)], check=True)
        (staging / "Applications").symlink_to("/Applications")
        subprocess.run(
            ["hdiutil", "create", "-volname", APP_NAME, "-srcfolder", str(staging), "-ov", "-format", "UDZO", str(dmg)],
            check=True,
        )
    return dmg


def linux_archive(app: Path, ver: str) -> Path:
    archive = DIST / f"{APP_NAME}-{ver}-linux-{platform.machine() or 'x86_64'}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(app, arcname=APP_NAME)
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Athena desktop app and installer.")
    parser.add_argument(
        "--no-kokoro",
        dest="with_kokoro",
        action="store_false",
        default=os.environ.get("ATHENA_BUNDLE_KOKORO", "1").strip().lower() not in ("0", "false", "no"),
        help="leave out the Kokoro voice pack (makes the app ~120 MB smaller)",
    )
    parser.add_argument(
        "--voice-source", type=Path, help="copy the Piper voice from this folder instead of downloading"
    )
    parser.add_argument("--skip-voices", action="store_true", help="use whatever is already in packaging/build/voices")
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--skip-installer", action="store_true")
    args = parser.parse_args(argv)
    ver = version()
    print(f"Building {APP_NAME} {ver} on {platform.platform()} with Python {platform.python_version()}")

    if not args.skip_voices:
        step("Voices")
        fetch_voices(kokoro=args.with_kokoro, source=args.voice_source)
    step("Icons")
    make_icons()
    step("PyInstaller")
    app = run_pyinstaller(kokoro=args.with_kokoro)
    print(f"  {app.relative_to(ROOT)}: {folder_size(app) / 1e6:.1f} MB")
    if not args.skip_smoke_test:
        step("Smoke test (--self-test-quick)")
        smoke_test(app)
    outputs: list[Path] = []
    if not args.skip_installer:
        step("Installer")
        if sys.platform == "win32":
            outputs.append(windows_installer(app, ver))
        elif sys.platform == "darwin":
            outputs.append(macos_dmg(app, ver))
        else:
            outputs.append(linux_archive(app, ver))
    step("Done")
    for path in outputs:
        print(f"  {path.relative_to(ROOT)}: {path.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
