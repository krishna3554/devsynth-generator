#!/usr/bin/env bash
set -euo pipefail

DATASET_FILE="chronic_kidney_disease_risk_2026.csv"

python3 gen.py --output "$DATASET_FILE"

python3 - <<'PY'
import csv

path = "chronic_kidney_disease_risk_2026.csv"

with open(path, newline="") as file:
    rows = list(csv.DictReader(file))

empty_cells = sum(value == "" for row in rows for value in row.values())
positive = sum(int(row["ckd_diagnosis"]) for row in rows)
prevalence = positive / len(rows) * 100

print(f"Rows: {len(rows)}")
print(f"Columns: {len(rows[0])}")
print(f"Empty cells: {empty_cells}")
print(f"CKD diagnosis prevalence: {prevalence:.1f}%")

if len(rows) != 9000:
    raise SystemExit("Expected 9000 rows")

if len(rows[0]) != 27:
    raise SystemExit("Expected 27 columns")

if empty_cells:
    raise SystemExit("Dataset has empty cells")
PY

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle command not found. Activate your virtual environment first."
  exit 1
fi

kaggle datasets version -p . -m "Regenerated synthetic CKD dataset"
