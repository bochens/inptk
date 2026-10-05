"""Build and verify an unsigned Windows x64 CLI folder, ZIP and per-user installer."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import struct
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_snapshot(destination):
    """Freeze the checkout, including explicitly allowed local source changes."""
    names = (
        run(
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            cwd=ROOT,
            capture_output=True,
        )
        .stdout.decode("utf-8")
        .split("\0")
    )
    hashes = {}
    for name in sorted(set(names) - {""}):
        source = ROOT / name
        if not source.is_file():
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = digest(target)
    return hashes


def copy_licenses(bundle):
    licenses = bundle / "licenses"
    licenses.mkdir()
    for distribution in importlib.metadata.distributions():
        for file in distribution.files or []:
            if "license" in str(file).lower() or file.name.lower().startswith("copying"):
                source = Path(distribution.locate_file(file))
                if source.is_file():
                    destination = licenses / distribution.metadata["Name"] / file
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    candidates = [
        Path(sys.base_prefix) / "LICENSE.txt",
        Path(sysconfig.get_path("stdlib")) / "LICENSE.txt",
        Path(sys.base_prefix) / "LICENSE_PYTHON.txt",
    ]
    python_license = next((path for path in candidates if path.is_file()), None)
    if python_license is None:
        raise RuntimeError("The build interpreter's Python license was not found")
    shutil.copy2(python_license, licenses / "Python.txt")


def find_iscc(value):
    if value:
        return value.resolve()
    on_path = shutil.which("ISCC.exe")
    if on_path:
        return Path(on_path)
    import os

    for variable, suffix in (
        ("ProgramFiles(x86)", "Inno Setup 6/ISCC.exe"),
        ("ProgramFiles", "Inno Setup 6/ISCC.exe"),
        ("LOCALAPPDATA", "Programs/Inno Setup 6/ISCC.exe"),
        ("ProgramFiles", "Inno Setup 7/ISCC.exe"),
    ):
        candidate = Path(os.environ.get(variable, "")) / suffix
        if candidate.is_file():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "windows")
    parser.add_argument("--work", type=Path, default=ROOT / "tmp" / "windows-package")
    parser.add_argument("--iscc", type=Path, help="Path to the Inno Setup compiler")
    parser.add_argument("--unsigned", action="store_true", help="Create test artifacts")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Include local source edits in the unsigned archive and label BUILD.json",
    )
    args = parser.parse_args()
    if (
        sys.platform != "win32"
        or platform.machine().lower() not in ("amd64", "x86_64")
        or struct.calcsize("P") != 8
        or sys.version_info[:2] != (3, 13)
    ):
        parser.error("Build with native Windows x64 Python 3.13")
    if not args.unsigned:
        parser.error("This builder currently requires --unsigned for test artifacts")
    compiler = find_iscc(args.iscc)
    if compiler is None or not compiler.is_file():
        parser.error("Install Inno Setup or supply --iscc PATH\\TO\\ISCC.exe")
    status = run(
        "git",
        "status",
        "--porcelain",
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    if status and not args.allow_dirty:
        parser.error("Commit source changes first, or use --unsigned --allow-dirty")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    if importlib.metadata.version("inptk") != version:
        parser.error("Install this checkout into the build environment first")
    commit = run(
        "git", "rev-parse", "HEAD", cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    output = args.out.resolve()
    base = f"inptk-{version}-windows-x64-unsigned"
    folder = output / base
    zip_path = output / f"{base}.zip"
    installer = output / f"{base}-setup.exe"
    for path in (folder, zip_path, installer):
        if path.exists():
            parser.error(f"Output already exists: {path}")
    args.work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="build-", dir=args.work.resolve()) as work:
        work = Path(work)
        source = work / "source"
        hashes = source_snapshot(source)
        run(
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onedir",
            "--console",
            "--noupx",
            "--name",
            "inptk",
            "--distpath",
            work / "dist",
            "--workpath",
            work / "build",
            "--specpath",
            work,
            "--paths",
            source / "src",
            "--copy-metadata",
            "inptk",
            source / "packaging/macos/entrypoint.py",
            cwd=source,
        )
        bundle = work / "dist" / "inptk"
        shutil.copy2(source / "LICENSE", bundle / "LICENSE")
        shutil.copy2(source / "packaging/windows/README.md", bundle / "README.md")
        archive = bundle / "SOURCE.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for name in hashes:
                stream.add(source / name, arcname=f"inptk-{version}/{name}", recursive=False)
        copy_licenses(bundle)
        (bundle / "BUILD.json").write_text(
            json.dumps(
                {
                    "version": version,
                    "commit": commit,
                    "source_dirty": bool(status),
                    "source_sha256": digest(archive),
                    "source_files_sha256": hashes,
                    "architecture": "x64",
                    "python": platform.python_version(),
                    "build_windows": platform.version(),
                    "unsigned": True,
                    "packages": {
                        d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        run(sys.executable, source / "packaging/check_executable.py", bundle / "inptk.exe")
        run(
            compiler,
            "/Qp",
            f"/DBundleDir={bundle}",
            f"/DAppVersion={version}",
            f"/O{work}",
            f"/F{base}-setup",
            source / "packaging/windows/inptk.iss",
        )
        shutil.copytree(bundle, folder)
        shutil.make_archive(
            str(zip_path.with_suffix("")), "zip", root_dir=output, base_dir=folder.name
        )
        shutil.copy2(work / installer.name, installer)
    print(f"Built executable: {folder / 'inptk.exe'}\nPortable ZIP: {zip_path}")
    print(f"Unsigned test installer: {installer}")


if __name__ == "__main__":
    main()
