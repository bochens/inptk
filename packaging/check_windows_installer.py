"""Check isolated Windows installation, update, busy-process protection and uninstall."""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
import winreg
from pathlib import Path

from check_executable import check

APP_ID = "{7F1A972A-7232-42C1-ABFD-A25452236977}_is1"


def installed():
    for view in (winreg.KEY_WOW64_32KEY, winreg.KEY_WOW64_64KEY):
        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                f"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{APP_ID}",
                0,
                winreg.KEY_READ | view,
            ):
                return True
        except FileNotFoundError:
            pass
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installer", help="Installer path or a pattern matching exactly one file")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Run on Windows")
    matches = glob.glob(args.installer)
    if len(matches) != 1:
        parser.error("Supply exactly one installer")
    if installed():
        parser.error("An INP-toolkit installation already exists; use a separate test account")
    installer = Path(matches[0]).resolve()
    with tempfile.TemporaryDirectory(prefix="inptk-install-check-") as directory:
        root = Path(directory)
        target = root / "INP toolkit é 冰"
        analysis = root / "retained analysis.inptk"
        analysis.write_text(
            "Analysis data outside the installation must survive.", encoding="utf-8"
        )
        common = ["/SP-", "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"]

        def setup(label, success=True):
            result = subprocess.run(
                [str(installer), *common, f"/DIR={target}", f"/LOG={root / (label + '.log')}"],
                timeout=180,
                check=False,
            )
            if (result.returncode == 0) != success:
                raise RuntimeError(
                    f"Installer {label} returned {result.returncode}: "
                    + (root / (label + ".log")).read_text(encoding="utf-8", errors="replace")
                )

        try:
            setup("install")
            assert installed()
            executable = target / "inptk.exe"
            assert executable.is_file() and (target / "_internal" / "python313.dll").is_file()
            check(executable)
            env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "CONDA"))}
            env["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
            with subprocess.Popen(
                [str(executable), "serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW,
            ) as process:
                process.stdin.write('{"id":1,"args":["capabilities"]}\n')
                process.stdin.flush()
                assert json.loads(process.stdout.readline())["status"] == "ok"
                setup("busy-update", success=False)
                assert process.poll() is None
                output, errors = process.communicate(
                    '{"id":2,"args":["capabilities"]}\n',
                    timeout=30,
                )
                assert process.returncode == 0, errors
                assert json.loads(output)["id"] == 2
            setup("update")
            assert (
                json.loads(
                    subprocess.check_output(
                        [str(executable), "capabilities"],
                        env=env,
                        text=True,
                        encoding="utf-8",
                    )
                )["status"]
                == "ok"
            )
        finally:
            uninstaller = target / "unins000.exe"
            if uninstaller.is_file():
                subprocess.run([str(uninstaller), *common], check=True, timeout=180)
                deadline = time.monotonic() + 15
                while target.exists() and time.monotonic() < deadline:
                    time.sleep(0.2)
            assert not installed(), "Test installation registry entry was not removed"
        assert not target.exists(), "Program files were not removed"
        assert analysis.read_text(encoding="utf-8").startswith("Analysis data")
    print("Installer passed install, active-process protection, update and uninstall checks.")


if __name__ == "__main__":
    main()
