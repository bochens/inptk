"""Build a standalone CLI installer for the current Mac architecture.

Use the dedicated build environment described in packaging/macos/README.md.
Unsigned packages are test artifacts. Public packages require Developer ID
application/installer certificates and a notarytool keychain profile.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "macos")
    parser.add_argument("--unsigned", action="store_true", help="Create a local test installer")
    parser.add_argument("--application-identity")
    parser.add_argument("--installer-identity")
    parser.add_argument("--notary-profile")
    args = parser.parse_args()
    if sys.platform != "darwin" or platform.machine() not in ("arm64", "x86_64"):
        parser.error("Build on an Apple Silicon or Intel Mac with a native Python environment")
    signing = [args.application_identity, args.installer_identity, args.notary_profile]
    if (args.unsigned and any(signing)) or (not args.unsigned and not all(signing)):
        parser.error("Choose --unsigned, or supply all three signing/notarization options")
    if run("git", "diff", "HEAD", "--", cwd=ROOT, capture_output=True).stdout:
        parser.error("Commit tracked changes before building a versioned source archive")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    if importlib.metadata.version("inptk") != version:
        parser.error("Install this checkout into the build environment first")
    architecture = platform.machine()
    commit = run(
        "git", "rev-parse", "HEAD", cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    suffix = "-unsigned" if args.unsigned else ""
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out.resolve() / f"inptk-{version}-macos-{architecture}{suffix}.pkg"
    if target.exists():
        parser.error(f"Output already exists: {target}")
    with tempfile.TemporaryDirectory(prefix="inptk-package-") as work:
        work = Path(work)
        # Use a folder bundle: a single-file executable would unpack SciPy at each startup.
        command = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onedir",
            "--console",
            "--name",
            "inptk",
            "--distpath",
            work / "dist",
            "--workpath",
            work / "build",
            "--specpath",
            work,
            "--paths",
            ROOT / "src",
            "--copy-metadata",
            "inptk",
        ]
        if not args.unsigned:
            command.extend(["--codesign-identity", args.application_identity])
        run(*command, ROOT / "packaging/macos/entrypoint.py", cwd=ROOT)
        bundle = work / "dist/inptk"
        shutil.copy2(ROOT / "LICENSE", bundle / "LICENSE")
        # Ship the exact tracked source alongside the executable and build instructions.
        run(
            "git",
            "archive",
            "--format=tar.gz",
            f"--prefix=inptk-{version}/",
            "-o",
            bundle / "SOURCE.tar.gz",
            "HEAD",
            cwd=ROOT,
        )
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
        shutil.copy2(Path(sysconfig.get_path("stdlib")) / "LICENSE.txt", licenses / "Python.txt")
        (bundle / "BUILD.json").write_text(
            json.dumps(
                {
                    "version": version,
                    "commit": commit,
                    "architecture": architecture,
                    "python": platform.python_version(),
                    "build_macos": platform.mac_ver()[0],
                    "minimum_macos": "14.0",
                    "unsigned": args.unsigned,
                    "packages": {
                        d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                    },
                },
                indent=2,
            )
            + "\n"
        )
        run(sys.executable, ROOT / "scripts/check_executable.py", bundle / "inptk", cwd=ROOT)
        identifier = "org.bochens.inptk"
        run(
            "/usr/bin/pkgbuild",
            "--root",
            bundle,
            "--identifier",
            identifier,
            "--version",
            version,
            "--install-location",
            "/Applications/INP-toolkit",
            "--ownership",
            "recommended",
            work / "component.pkg",
        )
        distribution = ET.Element("installer-gui-script", {"minSpecVersion": "2"})
        ET.SubElement(distribution, "title").text = "INP-toolkit"
        ET.SubElement(
            distribution,
            "options",
            {"customize": "never", "rootVolumeOnly": "true", "hostArchitectures": architecture},
        )
        versions = ET.SubElement(distribution, "allowed-os-versions")
        ET.SubElement(versions, "os-version", {"min": "14.0"})
        choices = ET.SubElement(distribution, "choices-outline")
        ET.SubElement(choices, "line", {"choice": "default"})
        choice = ET.SubElement(distribution, "choice", {"id": "default", "visible": "false"})
        ET.SubElement(choice, "pkg-ref", {"id": identifier})
        ET.SubElement(
            distribution, "pkg-ref", {"id": identifier, "version": version}
        ).text = "component.pkg"
        ET.ElementTree(distribution).write(work / "Distribution.xml", encoding="utf-8")
        command = [
            "/usr/bin/productbuild",
            "--distribution",
            work / "Distribution.xml",
            "--package-path",
            work,
        ]
        if not args.unsigned:
            command.extend(["--sign", args.installer_identity])
        run(*command, work / target.name)
        if not args.unsigned:
            run(
                "xcrun",
                "notarytool",
                "submit",
                work / target.name,
                "--keychain-profile",
                args.notary_profile,
                "--wait",
            )
            run("xcrun", "stapler", "staple", work / target.name)
            run("spctl", "--assess", "--type", "install", work / target.name)
        shutil.copy2(work / target.name, target)
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    target.with_suffix(".pkg.sha256").write_text(f"{digest}  {target.name}\n")
    print(
        f"Built {target}\nIcescopy executable after installation: /Applications/INP-toolkit/inptk"
    )


if __name__ == "__main__":
    main()
