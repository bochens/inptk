# Mac command-line distribution

Distribute INP-toolkit through GitHub Releases as a signed, notarized `.pkg`.
The installer puts the standalone command and its bundled libraries in:

```text
/Applications/INP-toolkit/inptk
```

Select that file in Icescopy's INP-toolkit executable preference. It is a command,
not a GUI app. No separate Python, Conda, package manager, or network connection
is needed to run an installed copy. Keep the `_internal` folder beside the
executable. Restart an existing toolkit process after installing an update.
The installer does not change shell settings or replace another `inptk` on PATH.
Terminal users can invoke the full path or add `/Applications/INP-toolkit` to PATH.

Build separate packages for Apple Silicon (`arm64`) and Intel (`x86_64`). The
installer requires macOS 14 or later and the matching architecture. Test on the
oldest supported system before advertising that minimum as verified; building
on a newer Mac alone does not verify older systems. The package contains the
Python runtime, NumPy, pandas and SciPy. The Python package remains separately
installable for notebook users.

## Build

Use a clean Python 3.13 environment to avoid bundling unrelated research tools:

```bash
python3.13 -m venv /tmp/inptk-build
/tmp/inptk-build/bin/python -m pip install -r packaging/macos/requirements.txt .
/tmp/inptk-build/bin/python scripts/build_macos.py --unsigned
```

Commit tracked changes before building. The builder includes the exact Git source
archive, dependency licenses, and `BUILD.json` with versions and commit identity.
It uses a temporary build directory and leaves only the installer and its SHA-256
checksum in `dist/macos`. Existing output packages are not overwritten.

The executable check starts the bundled process outside the source tree, uploads
raw sample/blank counts through JSON, and compares Average, joint MLE and their
uncertainty bounds with the Python API. It needs no installation or administrator
permission. This tests the bundled runtime; it does not test Installer upgrades
or Gatekeeper on a different Mac.

The `Mac installer` GitHub Actions workflow builds and checks both architectures.
Its unsigned artifacts are for testing; an Actions artifact is not a published
release. You can also run that workflow manually for a chosen commit or tag.

## Public release

Use a Mac with Developer ID Application and Developer ID Installer identities
in its keychain, plus a configured `notarytool` keychain profile:

```bash
/tmp/inptk-build/bin/python scripts/build_macos.py \
  --application-identity 'Developer ID Application: YOUR NAME (TEAMID)' \
  --installer-identity 'Developer ID Installer: YOUR NAME (TEAMID)' \
  --notary-profile inptk-release
```

This signs the executable and libraries, signs the installer, submits it to Apple,
attaches Apple's approval ticket, and checks the installer's Gatekeeper assessment.
Credentials stay in the keychain. An unsigned build is explicitly named
`-unsigned.pkg`; do not publish it as the normal end-user installer.

For each public version:

1. Merge reviewed changes and verify the version in `pyproject.toml` and
   `src/inptk/__init__.py`. Tag that exact commit, for example `v0.4.0`.
2. Build, sign and notarize both architecture packages from that tag.
3. Test installation and upgrade on clean Macs, then test Icescopy's executable
   chooser, JSON session and save/export with the installed command.
4. Attach the two `.pkg` files and checksums to the matching GitHub Release, with
   supported macOS versions and the executable path prominently stated.
5. Publish the wheel and source distribution for Python users when PyPI publishing
   is configured. Build these with `python -m build`; never ask normal Mac users
   to install the developer extras.

Do not use a single-file PyInstaller build: it extracts the numerical libraries
on startup. A folder bundle inside the installer avoids that repeated work.
There is no automatic updater; a later installer updates the same application
folder. To remove the installed program, remove `/Applications/INP-toolkit`;
analysis data are stored wherever the user chose, outside this folder.

References: [PyInstaller bundle behavior](https://pyinstaller.org/en/stable/operating-mode.html),
[Mac signing](https://pyinstaller.org/en/stable/feature-notes.html#macos-binary-code-signing),
[Apple distribution guidance](https://developer.apple.com/documentation/xcode/packaging-mac-software-for-distribution).
