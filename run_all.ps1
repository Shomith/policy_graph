$ErrorActionPreference = "Stop"

function Run($cmd) {
    Write-Host ">> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "Failed: $cmd" }
}

# data
Run "python make_data.py"
Run "python make_data_v2.py"
Run "python test_graph.py"

# baselines, then the GNN with both feature sets
Run "python -m pg.baselines data/policies_v2 data/labels_v2.csv"
Run "python -m pg.train data/policies_v2 data/labels_v2.csv base"
Run "python -m pg.train data/policies_v2 data/labels_v2.csv kinds"

Write-Host "Done. Model saved to model.pt" -ForegroundColor Green
Write-Host "Slower extras (about 15 minutes each on one CPU core):"
Write-Host "  python -m pg.ablate  data/policies_v2 data/labels_v2.csv 5"
Write-Host "  python -m pg.analyze data/policies_v2 data/labels_v2.csv kinds 5"
