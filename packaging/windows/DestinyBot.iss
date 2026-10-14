; Athena installer for Windows (Inno Setup 6).
;
; packaging/build.py compiles this after PyInstaller has built dist\Athena:
;   iscc /DAppVersion=0.1.0 /DSourceDir=<repo>\dist\Athena /DOutputDir=<repo>\dist DestinyBot.iss
;
; What it does:
; - installs for the current user only (no administrator password) into
;   %LOCALAPPDATA%\Programs\Athena, with a Start menu entry and a Desktop icon
;   (ticked by default), and offers to start Athena at the end;
; - checks for the Microsoft Edge WebView2 runtime that draws Athena's window.
;   Windows 10 and 11 normally have it; if it's missing, Setup offers to download
;   Microsoft's small official installer (about 2 MB) and runs it. If that is
;   declined or fails, Athena still works: it opens in a Microsoft Edge app window;
; - registers an uninstaller in Settings > Apps, which asks whether to also delete
;   the user's books, reading progress and settings (%LOCALAPPDATA%\Athena).

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef AppNumericVersion
  #define AppNumericVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\Athena"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif
#ifndef IconFile
  #define IconFile "..\icons\icon.ico"
#endif

#define AppName "Athena"
#define AppExe "Athena.exe"
; The data folder that destiny_bot.config.data_dir() uses on Windows.
#define DataDirName "Athena"
; Keep in sync with destiny_bot/desktop.py (APP_MUTEX, APP_ID).
#define AppMutexName "AthenaReaderAppMutex"
#define AppUserModelID "Athena.Reader"
#define WebView2Url "https://go.microsoft.com/fwlink/p/?LinkId=2124703"

[Setup]
AppId={{D304FF48-00F6-4FB5-A4FE-7C0ADAA89351}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppName}
AppComments=Reads your textbooks and class notes aloud. Private and offline.
VersionInfoVersion={#AppNumericVersion}
VersionInfoDescription={#AppName} Setup
VersionInfoProductName={#AppName}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableWelcomePage=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename={#AppName}-Setup-{#AppVersion}
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
WizardSizePercent=100
#ifdef WizardSmall
WizardSmallImageFile={#WizardSmall}
#endif
#ifdef WizardLarge
WizardImageFile={#WizardLarge}
#endif
AppMutex={#AppMutexName}
CloseApplications=yes
RestartApplications=no
ShowLanguageDialog=no
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
WelcomeLabel1=Welcome to {#AppName}
WelcomeLabel2={#AppName} reads your textbooks, slides and class notes aloud with a natural voice, and highlights each sentence as it goes.%n%nEverything stays private on this computer. Installing takes about a minute and doesn't need an administrator password.
FinishedHeadingLabel={#AppName} is ready
FinishedLabelNoIcons={#AppName} is installed.
FinishedLabel={#AppName} is installed. Open it any time from the {#AppName} icon on your desktop or in the Start menu.
ClickFinish=Click Finish to close Setup.

[CustomMessages]
english.ShortcutsGroup=Shortcuts:
english.DesktopIcon=Put an {#AppName} icon on my desktop
english.LaunchApp=Open {#AppName} now
english.WebView2Missing={#AppName} uses a free Microsoft component called WebView2 to show its window, and it isn't on this computer yet.%n%nDownload and install it now? It's about 2 MB, from Microsoft.%n%n(If you choose No, {#AppName} will open in a Microsoft Edge window instead.)
english.WebView2Installing=Installing Microsoft WebView2...
english.WebView2Failed=Microsoft WebView2 couldn't be installed right now. That's OK: {#AppName} will open in a Microsoft Edge window instead.
english.DeleteDataQuestion=Also delete your books, reading progress and settings?%n%nChoose No to keep them, so they are still there if you install {#AppName} again.
english.DeleteDataFailed=Some of your {#AppName} files couldn't be deleted. You can delete this folder yourself:%n%n%1

[Tasks]
Name: "desktopicon"; Description: "{cm:DesktopIcon}"; GroupDescription: "{cm:ShortcutsGroup}"

[InstallDelete]
; Updates replace the whole program folder so files from an older version never mix in.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Comment: "Reads your textbooks aloud"; AppUserModelID: "{#AppUserModelID}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Comment: "Reads your textbooks aloud"; AppUserModelID: "{#AppUserModelID}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchApp}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal"

[Code]
const
  WebView2ClientKey = 'Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';

var
  DownloadPage: TDownloadWizardPage;

function HasRuntimeVersion(const RootKey: Integer; const SubKey: String): Boolean;
var
  Version: String;
begin
  Result := RegQueryStringValue(RootKey, SubKey, 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0');
end;

{ The registry check Microsoft documents for the Evergreen WebView2 runtime. }
function WebView2Installed: Boolean;
begin
  Result := HasRuntimeVersion(HKLM32, 'SOFTWARE\' + WebView2ClientKey) or
            HasRuntimeVersion(HKCU, 'Software\' + WebView2ClientKey);
  if (not Result) and IsWin64 then
    Result := HasRuntimeVersion(HKLM64, 'SOFTWARE\' + WebView2ClientKey);
end;

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  Result := True;
end;

procedure InitializeWizard;
begin
  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing), SetupMessage(msgPreparingDesc), @OnDownloadProgress);
end;

procedure InstallWebView2;
var
  ResultCode: Integer;
begin
  DownloadPage.Clear;
  DownloadPage.Add('{#WebView2Url}', 'MicrosoftEdgeWebview2Setup.exe', '');
  DownloadPage.Show;
  try
    try
      DownloadPage.Download;
      DownloadPage.SetText(CustomMessage('WebView2Installing'), '');
      if not Exec(ExpandConstant('{tmp}\MicrosoftEdgeWebview2Setup.exe'), '/silent /install', '',
                  SW_HIDE, ewWaitUntilTerminated, ResultCode) or (ResultCode <> 0) then
      begin
        Log(Format('WebView2 installer failed with code %d', [ResultCode]));
        SuppressibleMsgBox(CustomMessage('WebView2Failed'), mbInformation, MB_OK, IDOK);
      end;
    except
      Log('WebView2 download failed: ' + GetExceptionMessage);
      SuppressibleMsgBox(CustomMessage('WebView2Failed'), mbInformation, MB_OK, IDOK);
    end;
  finally
    DownloadPage.Hide;
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = wpReady) and not WebView2Installed then
  begin
    Log('The WebView2 runtime is missing');
    if WizardSilent or
       (SuppressibleMsgBox(CustomMessage('WebView2Missing'), mbConfirmation, MB_YESNO, IDYES) = IDYES) then
      InstallWebView2;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\{#DataDirName}');
    { Silent uninstalls (e.g. by an update tool) always keep the user's data. }
    if DirExists(DataDir) and not UninstallSilent then
      if MsgBox(CustomMessage('DeleteDataQuestion'), mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        if not DelTree(DataDir, True, True, True) then
          MsgBox(FmtMessage(CustomMessage('DeleteDataFailed'), [DataDir]), mbInformation, MB_OK);
  end;
end;
