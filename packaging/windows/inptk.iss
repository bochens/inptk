; Build with scripts/build_windows.py. Keep this AppId stable for upgrades.
#ifndef BundleDir
  #error BundleDir must name the tested PyInstaller folder
#endif
#ifndef AppVersion
  #error AppVersion must match pyproject.toml
#endif

[Setup]
AppId={{7F1A972A-7232-42C1-ABFD-A25452236977}
AppName=INP-toolkit
AppVersion={#AppVersion}
AppPublisher=INP-toolkit contributors
AppPublisherURL=https://github.com/bochens/inptk
AppSupportURL=https://github.com/bochens/inptk/issues
DefaultDirName={localappdata}\Programs\INP-toolkit
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
MinVersion=10.0
LicenseFile={#BundleDir}\LICENSE
OutputBaseFilename=inptk-{#AppVersion}-windows-x64-unsigned-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=INP-toolkit {#AppVersion}
UninstallDisplayIcon={app}\inptk.exe
; Let Icescopy stop its own process; never close unrelated applications.
CloseApplications=no
RestartApplications=no
SetupLogging=yes

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Code]
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ToolkitPath: String;
  Stream: TFileStream;
begin
  Result := '';
  ToolkitPath := ExpandConstant('{app}\inptk.exe');
  if FileExists(ToolkitPath) then
  begin
    try
      Stream := TFileStream.Create(ToolkitPath, fmOpenReadWrite or fmShareExclusive);
      Stream.Free;
    except
      Result := 'Close the active INP-toolkit connection in Icescopy and any toolkit commands, then retry.';
    end;
  end;
end;
