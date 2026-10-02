#!/usr/bin/env bash
set -euo pipefail

python make_data.py
python make_data_v2.py
python test_graph.py
python -m pg.baselines data/policies_v2 data/labels_v2.csv
python -m pg.train data/policies_v2 data/labels_v2.csv
echo "Done. Model saved to model.pt"
