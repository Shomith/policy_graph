#!/usr/bin/env bash
set -euo pipefail

# data
python make_data.py
python make_data_v2.py
python test_graph.py

# baselines, then the GNN with both feature sets
python -m pg.baselines data/policies_v2 data/labels_v2.csv
python -m pg.train data/policies_v2 data/labels_v2.csv base
python -m pg.train data/policies_v2 data/labels_v2.csv kinds

echo "Done. Model saved to model.pt"
echo "Slower extras (about 15 minutes each on one CPU core):"
echo "  python -m pg.ablate  data/policies_v2 data/labels_v2.csv 5"
echo "  python -m pg.analyze data/policies_v2 data/labels_v2.csv kinds 5"
