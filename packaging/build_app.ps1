<#
.SYNOPSIS
  Build NanoScope for Windows (x86-64, onedir) and optionally a distributable ZIP.

.DESCRIPTION
  The whole procedure. Nothing else is needed and nothing is done by hand.

      .\packaging\build_app.ps1               full build -> dist\NanoScope\NanoScope.exe + ZIP
      .\packaging\build_app.ps1 -NoZip        stop after the build (fast iteration)
      .\packaging\build_app.ps1 -KeepCache    reuse PyInstaller's analysis cache
      .\packaging\build_app.ps1 -SkipSelfTest skip the post-build launch check

  Run from the REPO ROOT, in the environment the app will be built from:

      py -3.11 -m venv .venv
      .\.venv\Scripts\Activate.ps1
      pip install -e ".[dev]" pyinstaller pillow
      .\packaging\build_app.ps1

  PyInstaller does not cross-compile. This must run on Windows; the macOS build is
  packaging/build_app.sh and produces a .app instead.
#>
[CmdletBinding()]
param(
    [switch]$NoZip,
    [switch]$KeepCache,
    [switch]$SkipSelfTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $Here
Push-Location $Root
try {
    Write-Host "== NanoScope Windows build ==" -ForegroundColor Cyan
    Write-Host "python : $((Get-Command python).Source)"
    Write-Host "root   : $Root"

    # --- preflight ---------------------------------------------------------------------------
    # Fail on the missing piece with a sentence about it, rather than 200 lines into PyInstaller.
    python -c "import PyInstaller" 2>$null
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller is not installed in this environment: pip install pyinstaller" }
    python -c "import PySide6" 2>$null
    if ($LASTEXITCODE -ne 0) { throw "PySide6 is not installed: pip install -e `".[dev]`"" }
    python -c "import nanoscope" 2>$null
    if ($LASTEXITCODE -ne 0) { throw "nanoscope is not importable: run pip install -e . from $Root" }
    $AppVersion = python (Join-Path $Here "build_identity.py") --version
    if ($LASTEXITCODE -ne 0) { throw "Unable to read the NanoScope version" }
    python (Join-Path $Here "build_identity.py") --check-environment
    if ($LASTEXITCODE -ne 0) { throw "Runtime dependency preflight failed; use the validated release environment" }

    # --- icon --------------------------------------------------------------------------------
    # Checked in, so this is a no-op on a normal build; regenerated only if it is missing.
    $Ico = Join-Path $Here "resources\NanoScope.ico"
    if (-not (Test-Path $Ico)) {
        Write-Host "-- icon (regenerating)" -ForegroundColor Yellow
        python (Join-Path $Here "make_icon.py") --ico
    }

    # --- clean -------------------------------------------------------------------------------
    # A stale dist\ is how a build "succeeds" while shipping last week's DLLs.
    if (-not $KeepCache) {
        foreach ($d in @("build\NanoScope", "dist\NanoScope")) {
            if (Test-Path $d) { Write-Host "-- removing $d"; Remove-Item -Recurse -Force $d }
        }
    }

    # --- build -------------------------------------------------------------------------------
    Write-Host "-- PyInstaller" -ForegroundColor Cyan
    $specArgs = @("--noconfirm", "--distpath", "dist", "--workpath", "build")
    if ($KeepCache) { $specArgs += "--noconfirm" }
    python -m PyInstaller @specArgs (Join-Path $Here "NanoScope-win.spec")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

    $ExePath = "dist\NanoScope\NanoScope.exe"
    if (-not (Test-Path $ExePath)) { throw "build reported success but $ExePath does not exist" }

    $size = (Get-ChildItem -Recurse "dist\NanoScope" | Measure-Object -Property Length -Sum).Sum
    Write-Host ("-- built {0}  ({1:N0} MB)" -f $ExePath, ($size / 1MB)) -ForegroundColor Green

    # --- self-test ---------------------------------------------------------------------------
    # Launch the REAL exe with --self-test. This is what catches an over-broad prune: the build
    # is green, the exe exists, and it dies on a missing plugin or a module that was excluded.
    #
    # NanoScope.exe is a windowed program (console=False in the spec). PowerShell does not wait for
    # a windowed program and does not set $LASTEXITCODE from it, so `& $ExePath --self-test`
    # returned at once and the check read the PyInstaller run's exit code: it always "passed".
    # Start-Process -Wait waits, and redirecting its output is how a windowed exe's prints are seen.
    # (Keep this file ASCII: Windows PowerShell 5.1 reads a BOM-less script as ANSI, and a UTF-8
    # em dash decodes to a closing quote there, which breaks the string it is in.)
    if (-not $SkipSelfTest) {
        Write-Host "-- self-test" -ForegroundColor Cyan
        $SelfTestOut = Join-Path $Root "build\selftest.out.txt"
        $SelfTestErr = Join-Path $Root "build\selftest.err.txt"
        $proc = Start-Process -FilePath (Resolve-Path $ExePath).Path -ArgumentList "--self-test" `
            -Wait -PassThru -NoNewWindow `
            -RedirectStandardOutput $SelfTestOut -RedirectStandardError $SelfTestErr
        Get-Content $SelfTestOut, $SelfTestErr -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
        if ($proc.ExitCode -ne 0) { throw "self-test FAILED (exit $($proc.ExitCode)): the bundle is broken" }
        Write-Host "-- self-test passed" -ForegroundColor Green
    }

    # --- zip ---------------------------------------------------------------------------------
    if (-not $NoZip) {
        $Zip = "dist\NanoScope-$AppVersion-win64.zip"
        if (Test-Path $Zip) { Remove-Item -Force $Zip }
        Write-Host "-- zipping -> $Zip" -ForegroundColor Cyan
        Compress-Archive -Path "dist\NanoScope\*" -DestinationPath $Zip
        Write-Host ("-- wrote {0} ({1:N0} MB)" -f $Zip, ((Get-Item $Zip).Length / 1MB)) -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "Done. Run it with: .\$ExePath" -ForegroundColor Green
    Write-Host "NOTE: the exe is UNSIGNED. Windows SmartScreen will warn on first run;" -ForegroundColor Yellow
    Write-Host "      'More info' -> 'Run anyway'. Signing needs a code-signing certificate." -ForegroundColor Yellow
}
finally {
    Pop-Location
}
