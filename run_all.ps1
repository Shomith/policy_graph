$ErrorActionPreference = "Stop"

function Run($cmd) {
    Write-Host ">> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "Failed: $cmd" }
}

Run "python make_data.py"
Run "python make_data_v2.py"
Run "python test_graph.py"
Run "python -m pg.baselines data/policies_v2 data/labels_v2.csv"
Run "python -m pg.train data/policies_v2 data/labels_v2.csv"
Write-Host "Done. Model saved to model.pt" -ForegroundColor Green
