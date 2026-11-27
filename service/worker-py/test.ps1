<#
  test.ps1 — one-command demo of the lipsync service (no Rust, no queue).

  It analyzes a video once, then synthesizes a section in one or more languages,
  reusing the same selection (exactly the "test many languages" workflow). Output
  clips land in a folder; the selection is released at the end.

  Examples:
    .\test.ps1
    .\test.ps1 -Languages hi,es,fr
    .\test.ps1 -Video "C:\clips\movie.mp4" -Section 1 -Languages hi -Steps 20
    .\test.ps1 -Keep          # keep the selection cached for more manual tests

  Parameters:
    -Video       Path to the input video (default: the bundled sample).
    -Languages   One or more target languages, comma-separated (default: hi).
    -Section     Which section id to render (default: 1).
    -Steps       Inference steps; 12 = quick test, 20-30 = quality (default: 12).
    -OutDir      Where to write the clips (default: .\service-test-out).
    -Keep        Do not release the selection at the end.
#>
param(
    [string]   $Video     = "$PSScriptRoot\..\..\LatentSync-trial\trial_inputs\sample_ref.mp4",
    [string[]] $Languages = @("hi"),
    [int]      $Section   = 1,
    [int]      $Steps     = 12,
    [string]   $OutDir    = "$PSScriptRoot\service-test-out",
    [switch]   $Keep
)

$py  = "$PSScriptRoot\..\..\LatentSync-trial\.venv\Scripts\python.exe"
$ctl = "$PSScriptRoot\servicectl.py"

function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $py))    { Fail "Python venv not found at $py" }
if (-not (Test-Path $ctl))   { Fail "servicectl.py not found at $ctl" }
if (-not (Test-Path $Video)) { Fail "Input video not found: $Video" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$manifestFile = Join-Path $OutDir "manifest.json"

# ---- Step 1: analyze (once) ---------------------------------------------------
Write-Host ""
Write-Host "[1/2] Analyzing: $Video" -ForegroundColor Cyan
& $py $ctl analyze $Video --json-out $manifestFile
if ($LASTEXITCODE -ne 0) { Fail "analyze failed (exit $LASTEXITCODE)" }

$manifest  = Get-Content $manifestFile -Raw | ConvertFrom-Json
$selection = $manifest.selection_id
$sec       = $manifest.sections | Where-Object { $_.id -eq $Section } | Select-Object -First 1
if (-not $selection) { Fail "No selection_id in manifest." }
if (-not $sec)       { Fail "Section $Section not found. Available: $(($manifest.sections.id) -join ', ')" }

Write-Host ""
Write-Host "  selection_id : $selection" -ForegroundColor Green
Write-Host "  sections     : $($manifest.sections.Count)" -ForegroundColor Green
Write-Host "  section $Section speakers : $($sec.speakers -join ', ')" -ForegroundColor Green

# ---- Step 2: synthesize (one clip per language, same selection) ---------------
$made = @()
foreach ($lang in $Languages) {
    $out = Join-Path $OutDir ("section{0}_{1}.mp4" -f $Section, $lang)
    Write-Host ""
    Write-Host "[2/2] Synthesizing section $Section -> $lang  (steps $Steps)" -ForegroundColor Cyan
    & $py $ctl synthesize $selection $Section $lang --steps $Steps --out $out
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  synthesize '$lang' FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        continue
    }
    Write-Host "  -> $out" -ForegroundColor Green
    $made += $out
}

# ---- cleanup ------------------------------------------------------------------
if (-not $Keep) {
    & $py $ctl release $selection | Out-Null
    Write-Host ""
    Write-Host "Released selection $selection (pass -Keep to retain it for more tests)." -ForegroundColor DarkGray
} else {
    Write-Host ""
    Write-Host "Selection kept: $selection  (release later with:  python servicectl.py release $selection)" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Done. $($made.Count) clip(s) in: $OutDir" -ForegroundColor Cyan
foreach ($m in $made) { Write-Host "   $m" }
