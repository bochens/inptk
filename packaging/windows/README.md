# Windows command-line distribution

`packaging/build_windows.py` builds a standalone Windows x64 command, portable ZIP
and per-user Inno Setup installer. Python calculations, CLI commands, JSON
protocol 2 and saved format 4 remain identical to the Mac/Python distribution.

The executable is a command-line program used directly by Icescopy. No separate
Python or Conda installation is needed to run the bundle. After installation,
select `%LOCALAPPDATA%\Programs\INP-toolkit\inptk.exe` in Icescopy. For a portable
copy, extract the entire ZIP and select its `inptk.exe`. Keep `_internal` beside
the executable; copying only the launcher will not work.

## Intended installation

- Start with native Windows x64. Do not advertise native ARM64 until separately built/tested.
- Bundle Python and numerical libraries with PyInstaller's `--onedir --console` mode.
- Package the entire folder with Inno Setup as a per-user installation.
- Use `DefaultDirName={localappdata}\Programs\INP-toolkit` and `PrivilegesRequired=lowest`.
- Icescopy selects `%LOCALAPPDATA%\Programs\INP-toolkit\inptk.exe`.
- No separate Python/Conda installation, GUI, Windows service or elevated analysis process.
- Preserve the `_internal` directory. Do not distribute only the small launcher EXE.
- Do not change PATH by default. Keep installation/uninstallation separate from analysis data.

Inno Setup supports installation without administrator rights through
[PrivilegesRequired=lowest](https://jrsoftware.org/ishelp/topic_setup_privilegesrequired.htm).
Use a stable installer AppId and destination for upgrades. Stop/restart Icescopy's
active toolkit process before replacing DLLs; do not silently kill unrelated processes.

## Build

Use native Windows x64 Python 3.13 and Inno Setup 6.3 or newer. The Windows builder
reuses the pinned requirements and CLI entrypoint from `packaging/macos`; these
Python packages are cross-platform. In PowerShell, from the repository root:

```powershell
py -3.13 -m venv tmp/windows-build-env
tmp/windows-build-env/Scripts/python.exe -m pip install -r packaging/macos/requirements.txt . pytest
tmp/windows-build-env/Scripts/python.exe -m pytest -q
tmp/windows-build-env/Scripts/python.exe packaging/build_windows.py --unsigned
tmp/windows-build-env/Scripts/python.exe packaging/check_windows_installer.py 'dist/windows/*-setup.exe'
```

Use `--iscc 'PATH\TO\ISCC.exe'` if the compiler is outside its standard installation
directory. Commit source changes before building. For a local test of uncommitted
packaging work, add `--allow-dirty`; the builder includes the exact source snapshot
and file hashes and labels `source_dirty` in `BUILD.json`. It builds from that
snapshot, preserving the relationship between the executable and shipped source.
Only unsigned test artifacts are currently supported by this builder.

Outputs in `dist/windows` are:

- `inptk-VERSION-windows-x64-unsigned/inptk.exe` plus `_internal`, source and licenses.
- `inptk-VERSION-windows-x64-unsigned.zip` for portable use.
- `inptk-VERSION-windows-x64-unsigned-setup.exe` for per-user installation.
- SHA-256 checksums for the ZIP and installer.

Build scratch stays in `tmp/windows-package` and is cleaned on completion.
Existing output artifacts are not overwritten. `SOURCE.tar.gz` contains the
exact source; `BUILD.json` records the commit, source hashes, architecture, Python
and installed dependency versions. Binaries remain ignored by Git.

## Verification

The builder runs `packaging/check_executable.py` before producing the installer.
It compares Average, joint MLE and uncertainty with the Python API, then checks
saved stages, explicit blanks, native CSV and `.icescopy` archives, spaced/Unicode
paths, JSON import, named curves, release and recovery after failed requests.
It runs outside the source tree with Python/Conda environment variables removed
and, on Windows, only Windows system files on PATH.

`check_windows_installer.py` installs into an isolated temporary directory, runs
those executable checks, verifies an update is blocked while `serve` is active,
then updates and uninstalls. Analysis files outside the installation survive.
It refuses to run if this user already has an INP-toolkit installation, avoiding
replacement of an existing installation's registration. Use a separate Windows
test account in that case. Keep the full-suite and installer-check logs with the release validation records.

These local checks do not establish downloaded-installer behavior on a separate
clean PC, Windows signing/SmartScreen, or Icescopy's executable chooser. Complete
those release checks before public distribution.

Keep `--console` in the build. Icescopy should start it using QProcess with pipes
without opening a separate terminal. Verify that behavior on Windows rather than
changing the CLI into a windowed app, which would remove standard streams.

Use the in-memory protocol in [the README](../../README.md#interactive-application-clients): upload counts/metadata once,
calculate into named references, request summary ranges and plot columns, save
only when asked, and release superseded results.
Final cumulative tables report only the sample freezing interval. Use those rows
for plots and export; counts outside that interval remain available to the fit. No Windows-specific estimator
or temporary-CSV calculation loop should be introduced.

## Distribution

Attach a signed `inptk-VERSION-windows-x64-setup.exe` and checksum to the same
GitHub Release as the Mac packages. Use Windows code signing when credentials are
available, and test the downloaded installer on a clean PC. Keep unsigned builds
explicitly marked as test installers; do not claim signing or SmartScreen behavior
has been verified before testing it. Keep version numbers aligned with the Mac
package and Python distribution. A portable ZIP of the full folder can be a
secondary option; the installer is the normal route for Icescopy users.
