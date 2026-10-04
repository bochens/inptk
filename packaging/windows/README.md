# Windows release handoff for the PC

Build a standalone Windows command and installer from the merged INP-toolkit
source. The Mac package has been built locally; Windows has not been built or
tested in this work. Keep the Python calculations, CLI commands, JSON protocol
and saved format identical on both systems.

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

## Build and verify

Use a clean native Windows Python 3.13 environment. The pinned build dependencies
are currently in `packaging/macos/requirements.txt`; they are not Mac-only Python
packages. Move common dependencies/entrypoint to a shared packaging location if
useful when implementing the Windows builder, updating the Mac references too.
The existing `packaging/macos/entrypoint.py` calls `inptk.cli.main` without GUI code.

Implement `scripts/build_windows.py` and an Inno Setup script with these steps:

1. Install the pinned build requirements and the current checkout into the build environment.
2. Build the CLI with `--onedir --console --name inptk`, using the shared entrypoint.
   Do not use `--windowed`: the Icescopy connection needs stdin and stdout.
3. Run `python scripts/check_executable.py PATH\TO\inptk.exe` against the frozen
   executable. This checks direct JSON upload, Average, joint MLE and uncertainty
   against the Python API from a clean working directory.
4. Run the package's full tests on Windows and verify saved-step commands, CSV and
   .icescopy input, Unicode/spaced paths, explicit blanks, release, and error recovery.
5. Keep the tested installer for upload to the GitHub Release.
   Keep temporary files in a dedicated build directory; do not commit binaries.
6. Include the exact source archive, package/Python licenses, dependency versions,
   architecture and commit identity, following `scripts/build_macos.py`.
7. Test installation, update and uninstall on a machine without Python, including
   Icescopy's executable chooser and a long-lived `serve` session.

Keep `--console` in the build. Icescopy should start it using QProcess with pipes
without opening a separate terminal. Verify that behavior on Windows rather than
changing the CLI into a windowed app, which would remove standard streams.

Use the in-memory protocol in [the README](../../README.md#interactive-application-clients): upload counts/metadata once,
calculate into named references, request summary ranges and plot columns, save
only when asked, and release superseded results. No Windows-specific estimator
or temporary-CSV calculation loop should be introduced.

## Distribution

Attach a signed `inptk-VERSION-windows-x64-setup.exe` to the same
GitHub Release as the Mac packages. Use Windows code signing when credentials are
available, and test the downloaded installer on a clean PC. Keep unsigned builds
explicitly marked as test installers; do not claim signing or SmartScreen behavior
has been verified before testing it. Keep version numbers aligned with the Mac
package and Python distribution. A portable ZIP of the full folder can be a
secondary option; the installer is the normal route for Icescopy users.
