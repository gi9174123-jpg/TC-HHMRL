param(
    [string]$PythonBin = "python",
    [string]$CfgPath = "configs/default.yaml",
    [int]$MetaIters = 5,
    [string]$Device = "auto",
    [string]$CsvPath = "logs/meta_run/metrics.csv"
)

$ErrorActionPreference = "Stop"
$env:MPLBACKEND = if ($env:MPLBACKEND) { $env:MPLBACKEND } else { "Agg" }

& $PythonBin -m pip install -r requirements.txt
& $PythonBin -m scripts.train_meta --cfg $CfgPath --meta-iters $MetaIters --device $Device
& $PythonBin -m scripts.eval --cfg $CfgPath --n-tasks 2 --episodes-per-task 1 --device $Device
& $PythonBin -m scripts.plot_results --csv $CsvPath

Write-Host "Repro run complete."
