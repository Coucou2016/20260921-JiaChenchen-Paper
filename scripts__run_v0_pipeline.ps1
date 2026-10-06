# run_v0_pipeline.ps1
# One-shot, unattended V0 pipeline for HydroGeo-SRNO:
#   1) full training (adopts an already-running trainer, otherwise starts / resumes one)
#   2) evaluation on val+test for the trained checkpoint and bilinear/nearest baselines
#   3) final summary appended to outputs/v0_10m2m_hmax/pipeline_status.txt
#
# Robustness:
#   - If a healthy trainer is already running (train.pid alive + train_fixed.py), it is
#     ADOPTED; no second training process is ever started.
#   - If training exits before the target epoch count, it is retried ONCE from the latest
#     checkpoint (unless an early-stop marker is found).
#   - Evaluation ALWAYS runs afterwards, even if training failed, so partial results are
#     recorded.
#
# Everything is logged to outputs/v0_10m2m_hmax/pipeline.log

[CmdletBinding()]
param(
    [int]$PollSeconds = 60,
    [int]$HeartbeatSeconds = 300
)

$ErrorActionPreference = 'Continue'

$PSScriptRoot_ = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Root      = Split-Path -Parent $PSScriptRoot_
$OutDir    = Join-Path $Root 'outputs\v0_10m2m_hmax'
$MetricsDir= Join-Path $OutDir 'metrics'
$PipelineLog = Join-Path $OutDir 'pipeline.log'
$StatusFile  = Join-Path $OutDir 'pipeline_status.txt'
$TrainPidFile= Join-Path $OutDir 'train.pid'
$PipelinePidFile = Join-Path $OutDir 'pipeline.pid'
$ConfigPath  = Join-Path $Root 'configs\v0_10m2m_hmax.yaml'

$Python = $env:V0_PYTHON
if (-not $Python -or -not (Test-Path $Python)) {
    if (Test-Path 'E:\Miniconda3\python.exe') { $Python = 'E:\Miniconda3\python.exe' }
    else { $Python = 'python' }
}

New-Item -ItemType Directory -Force -Path $OutDir     | Out-Null
New-Item -ItemType Directory -Force -Path $MetricsDir | Out-Null

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
function Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Write-Host $line
    try { Add-Content -LiteralPath $PipelineLog -Value $line -Encoding UTF8 } catch {}
}

# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
function Get-ConfigEpochs {
    if (-not (Test-Path $ConfigPath)) { return 300 }
    $txt = Get-Content -LiteralPath $ConfigPath -Raw -ErrorAction SilentlyContinue
    if ($txt -and $txt -match '(?m)^\s*epochs:\s*(\d+)') { return [int]$Matches[1] }
    return 300
}

function Get-LastEpoch {
    $h = Join-Path $OutDir 'history.jsonl'
    if (-not (Test-Path $h)) { return 0 }
    $last = Get-Content -LiteralPath $h -Tail 1 -ErrorAction SilentlyContinue
    if (-not $last) { return 0 }
    try { return [int]((ConvertFrom-Json $last).epoch) } catch { return 0 }
}

function Get-HistoryCount {
    $h = Join-Path $OutDir 'history.jsonl'
    if (-not (Test-Path $h)) { return 0 }
    try { return @(Get-Content -LiteralPath $h -ErrorAction SilentlyContinue).Count } catch { return 0 }
}

function Get-BestSummary {
    $p = Join-Path $OutDir 'best_summary.json'
    if (-not (Test-Path $p)) { return $null }
    try { return ConvertFrom-Json (Get-Content -LiteralPath $p -Raw) } catch { return $null }
}

function Test-LogMarker {
    param([string[]]$Patterns)
    foreach ($f in @('train.stdout.log','train.retry.stdout.log','train.log','console.log')) {
        $p = Join-Path $OutDir $f
        if (Test-Path $p) {
            foreach ($pat in $Patterns) {
                try {
                    if (Select-String -LiteralPath $p -Pattern $pat -SimpleMatch -Quiet -ErrorAction SilentlyContinue) {
                        return $true
                    }
                } catch {}
            }
        }
    }
    return $false
}

function Test-TrainerCommandLine {
    param([int]$ProcessId)
    try {
        $ci = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
        if ($ci -and $ci.CommandLine -and $ci.CommandLine -match 'train_fixed\.py') { return $true }
    } catch {}
    return $false
}

function Get-RunningTrainer {
    if (-not (Test-Path $TrainPidFile)) { return $null }
    $raw = Get-Content -LiteralPath $TrainPidFile -Raw -ErrorAction SilentlyContinue
    if (-not $raw) { return $null }
    $tpid = 0
    if (-not [int]::TryParse($raw.Trim(), [ref]$tpid)) { return $null }
    $p = Get-Process -Id $tpid -ErrorAction SilentlyContinue
    if ($null -eq $p) { return $null }
    if ($p.ProcessName -notmatch 'python') { return $null }
    if (-not (Test-TrainerCommandLine -ProcessId $tpid)) { return $null }
    return $p
}

function Get-MetricValue {
    param([string]$Path, [string]$Key)
    if (-not (Test-Path $Path)) { return $null }
    try {
        $j = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        if ($j.PSObject.Properties.Name -contains $Key) { return $j.$Key }
    } catch {}
    return $null
}

function Format-Num {
    param($Value)
    if ($null -eq $Value) { return 'n/a' }
    try { return ('{0:F4}' -f [double]$Value) } catch { return "$Value" }
}

# ----------------------------------------------------------------------------
# Trainer start / wait
# ----------------------------------------------------------------------------
function Start-TrainerProcess {
    param([string]$Tag)

    $lastPt = Join-Path $OutDir 'last.pt'
    $argList = @('-u', 'scripts/train_fixed.py', '--config', 'configs/v0_10m2m_hmax.yaml')
    if (Test-Path $lastPt) {
        $argList += @('--resume', ('"' + (Resolve-Path $lastPt).Path + '"'))
    }

    $outLog = Join-Path $OutDir ("train.{0}.stdout.log" -f $Tag)
    $errLog = Join-Path $OutDir ("train.{0}.stderr.log" -f $Tag)

    Log ("Starting trainer ({0}): {1} {2}" -f $Tag, $Python, ($argList -join ' '))
    $proc = Start-Process -FilePath $Python -ArgumentList $argList -WorkingDirectory $Root `
        -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog
    Set-Content -LiteralPath $TrainPidFile -Value $proc.Id -Encoding ASCII
    Log ("Trainer ({0}) started with PID {1}; stdout={2}" -f $Tag, $proc.Id, $outLog)
    return $proc
}

function Wait-ForTrainer {
    param([int]$ProcessId, [string]$Label)

    $lastHeartbeat = Get-Date
    $lastEpochSeen = -1
    while ($true) {
        $p = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if ($null -eq $p) { break }

        $epoch = Get-LastEpoch
        $now = Get-Date
        if ($epoch -ne $lastEpochSeen -or ($now - $lastHeartbeat).TotalSeconds -ge $HeartbeatSeconds) {
            $bs = Get-BestSummary
            $bestCsi = if ($bs) { Format-Num $bs.best_csi } else { 'n/a' }
            Log ("[{0}] alive pid={1} last_epoch={2} best_csi={3}" -f $Label, $ProcessId, $epoch, $bestCsi)
            $lastEpochSeen = $epoch
            $lastHeartbeat = $now
        }
        Start-Sleep -Seconds $PollSeconds
    }
    Log ("[{0}] pid={1} is no longer running; last_epoch={2}" -f $Label, $ProcessId, (Get-LastEpoch))
}

function Get-ProcExitCode {
    param($Proc)
    if ($null -eq $Proc) { return $null }
    try {
        $Proc.Refresh()
        return $Proc.ExitCode
    } catch {
        return $null
    }
}

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
function Write-Summary {
    param(
        [hashtable]$TrainInfo,
        [array]$EvalResults
    )

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add('')
    $lines.Add('================================================================')
    $lines.Add(("[V0 PIPELINE STATUS] {0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')))
    $lines.Add(("config      : {0}" -f $ConfigPath))
    $lines.Add(("python      : {0}" -f $Python))
    $lines.Add('----------------------------------------------------------------')
    $lines.Add('STAGE 1 - training')
    $lines.Add(("  adopted_existing_trainer : {0}" -f $TrainInfo.adopted))
    $lines.Add(("  adopted_pid              : {0}" -f $TrainInfo.adoptedPid))
    $lines.Add(("  attempt1_exit_code       : {0}" -f $TrainInfo.exit1))
    $lines.Add(("  attempt2_exit_code       : {0}" -f $TrainInfo.exit2))
    $lines.Add(("  target_epochs            : {0}" -f $TrainInfo.targetEpochs))
    $lines.Add(("  epochs_completed         : {0}" -f $TrainInfo.epochsCompleted))
    $lines.Add(("  history_rows             : {0}" -f $TrainInfo.historyRows))
    $lines.Add(("  best_csi                 : {0}" -f $TrainInfo.bestCsi))
    $lines.Add(("  best_rmse_wet            : {0}" -f $TrainInfo.bestRmseWet))
    $lines.Add(("  checkpoint               : {0}" -f $TrainInfo.checkpoint))
    $lines.Add('----------------------------------------------------------------')
    $lines.Add('STAGE 2 - evaluation')
    foreach ($r in $EvalResults) {
        $lines.Add(("  {0,-14} {1,-5} exit={2,-4} file={3}" -f $r.model, $r.split, $r.exit, $r.file))
        $lines.Add(("      CSI_005={0}  F1_005={1}  RMSE_wet={2}  MAE_wet={3}  CSI_030={4}  CSI_100={5}" -f `
            (Format-Num $r.csi005), (Format-Num $r.f1_005), (Format-Num $r.rmseWet), `
            (Format-Num $r.maeWet), (Format-Num $r.csi030), (Format-Num $r.csi100)))
    }
    $lines.Add('----------------------------------------------------------------')
    $lines.Add(("  metric dir : {0}" -f $MetricsDir))
    $lines.Add(("  pipeline   : {0}" -f $PipelineLog))
    $lines.Add('================================================================')

    $block = ($lines -join [Environment]::NewLine)
    try {
        Add-Content -LiteralPath $StatusFile -Value $block -Encoding UTF8
        Log ("Summary appended to {0}" -f $StatusFile)
    } catch {
        Log ("ERROR appending summary: {0}" -f $_)
    }
}

# ============================================================================
# MAIN
# ============================================================================
Log '=============================================================='
Log ("V0 pipeline start (pid={0}) root={1}" -f $PID, $Root)

# Single-instance guard
if (Test-Path $PipelinePidFile) {
    $rawPid = Get-Content -LiteralPath $PipelinePidFile -Raw -ErrorAction SilentlyContinue
    $prev = 0
    if ($rawPid -and [int]::TryParse($rawPid.Trim(), [ref]$prev)) {
        # Ignore a stale PID file that points at ourselves.
        if ($prev -ne $PID) {
            $prevProc = Get-Process -Id $prev -ErrorAction SilentlyContinue
            if ($prevProc) {
                Log ("Another pipeline instance appears alive (pid={0}); exiting to avoid duplicates." -f $prev)
                return
            }
        }
    }
}
Set-Content -LiteralPath $PipelinePidFile -Value $PID -Encoding ASCII

$targetEpochs = Get-ConfigEpochs
Log ("Target epochs = {0}; output dir = {1}" -f $targetEpochs, $OutDir)

$adopted = $false
$adoptedPid = 'n/a'
$exit1 = 'n/a'
$exit2 = 'n/a'

$running = Get-RunningTrainer
if ($running) {
    $adopted = $true
    $adoptedPid = $running.Id
    Log ("Healthy trainer already running: PID {0} (epoch {1}/{2}). Adopting it; NOT starting a second process." -f `
        $running.Id, (Get-LastEpoch), $targetEpochs)
    Wait-ForTrainer -ProcessId $running.Id -Label 'adopt'
} else {
    Log 'No healthy trainer found; starting training (resume from latest checkpoint if present).'
    $proc = Start-TrainerProcess -Tag 'attempt1'
    Wait-ForTrainer -ProcessId $proc.Id -Label 'attempt1'
    $exit1 = Get-ProcExitCode -Proc $proc
    Log ("Training attempt 1 exit code = {0}" -f $exit1)
}

# Is training complete?
$earlyStop = Test-LogMarker -Patterns @('early stop', 'reached max_steps', 'done. checkpoints')
$lastEpoch = Get-LastEpoch
if (($lastEpoch -ge $targetEpochs) -or $earlyStop) {
    Log ("Training complete: last_epoch={0}/{1}, early_stop_marker={2}" -f $lastEpoch, $targetEpochs, $earlyStop)
} else {
    Log ("Training incomplete: last_epoch={0}/{1}. Retrying ONCE from latest checkpoint." -f $lastEpoch, $targetEpochs)
    $proc = Start-TrainerProcess -Tag 'attempt2'
    Wait-ForTrainer -ProcessId $proc.Id -Label 'attempt2'
    $exit2 = Get-ProcExitCode -Proc $proc
    Log ("Training attempt 2 exit code = {0}" -f $exit2)
}

# Locate checkpoint for evaluation (prefer best_csi)
$bestCkpt = $null
foreach ($cand in @('best_csi.pt', 'best_rmse_wet.pt', 'last.pt')) {
    $p = Join-Path $OutDir $cand
    if (Test-Path $p) { $bestCkpt = $p; break }
}
if ($bestCkpt) { Log ("Using checkpoint for evaluation: {0}" -f $bestCkpt) }
else { Log 'WARNING: no checkpoint found; trained-model evaluation will be skipped.' }

# ---------------------------------------------------------------- STAGE 2: eval
$evalSpecs = @()
if ($bestCkpt) {
    $evalSpecs += @{ model = 'hydrogeo_srno'; split = 'val';  ckpt = $bestCkpt; out = (Join-Path $MetricsDir 'hydrogeo_srno_val.json') }
    $evalSpecs += @{ model = 'hydrogeo_srno'; split = 'test'; ckpt = $bestCkpt; out = (Join-Path $MetricsDir 'hydrogeo_srno_test.json') }
}
$evalSpecs += @{ model = 'bilinear'; split = 'val';  ckpt = $null; out = (Join-Path $MetricsDir 'bilinear_val.json') }
$evalSpecs += @{ model = 'bilinear'; split = 'test'; ckpt = $null; out = (Join-Path $MetricsDir 'bilinear_test.json') }
$evalSpecs += @{ model = 'nearest';  split = 'val';  ckpt = $null; out = (Join-Path $MetricsDir 'nearest_val.json') }
$evalSpecs += @{ model = 'nearest';  split = 'test'; ckpt = $null; out = (Join-Path $MetricsDir 'nearest_test.json') }

$evalResults = @()
foreach ($spec in $evalSpecs) {
    $argList = @('-u', 'scripts/evaluate.py', '--config', 'configs/v0_10m2m_hmax.yaml',
                 '--model', $spec.model, '--split', $spec.split, '--out', ('"' + $spec.out + '"'))
    if ($spec.ckpt) { $argList += @('--ckpt', ('"' + $spec.ckpt + '"')) }

    Log ("Eval start: model={0} split={1}" -f $spec.model, $spec.split)
    $code = $null
    try {
        $p = Start-Process -FilePath $Python -ArgumentList $argList -WorkingDirectory $Root `
            -WindowStyle Hidden -PassThru -Wait `
            -RedirectStandardOutput (Join-Path $OutDir ("eval.{0}.{1}.stdout.log" -f $spec.model, $spec.split)) `
            -RedirectStandardError  (Join-Path $OutDir ("eval.{0}.{1}.stderr.log" -f $spec.model, $spec.split))
        $code = Get-ProcExitCode -Proc $p
    } catch {
        Log ("Eval exception ({0}/{1}): {2}" -f $spec.model, $spec.split, $_)
    }
    Log ("Eval done: model={0} split={1} exit={2}" -f $spec.model, $spec.split, $code)

    $evalResults += [pscustomobject]@{
        model   = $spec.model
        split   = $spec.split
        exit    = $code
        file    = $spec.out
        csi005  = Get-MetricValue -Path $spec.out -Key 'CSI_005'
        f1_005  = Get-MetricValue -Path $spec.out -Key 'F1_005'
        rmseWet = Get-MetricValue -Path $spec.out -Key 'RMSE_wet'
        maeWet  = Get-MetricValue -Path $spec.out -Key 'MAE_wet'
        csi030  = Get-MetricValue -Path $spec.out -Key 'CSI_030'
        csi100  = Get-MetricValue -Path $spec.out -Key 'CSI_100'
    }
}

# ---------------------------------------------------------------- STAGE 3: summary
$bs = Get-BestSummary
$trainInfo = @{
    adopted         = $adopted
    adoptedPid      = $adoptedPid
    exit1           = $exit1
    exit2           = $exit2
    targetEpochs    = $targetEpochs
    epochsCompleted = (Get-LastEpoch)
    historyRows     = (Get-HistoryCount)
    bestCsi         = if ($bs) { Format-Num $bs.best_csi } else { 'n/a' }
    bestRmseWet     = if ($bs) { Format-Num $bs.best_rmse_wet } else { 'n/a' }
    checkpoint      = if ($bestCkpt) { $bestCkpt } else { 'none' }
}
Write-Summary -TrainInfo $trainInfo -EvalResults $evalResults

Log 'V0 pipeline finished.'
