import csv
import numpy as np
from baseline_ecmp import run_baseline

# Thay bằng số lấy từ eval_gnn.py
GNN_MLU = 0.72     # ví dụ
GNN_AVG = 0.51     # ví dụ

ecmp_mlu = run_baseline("ISP-only")
random_mlu = run_baseline("random")

with open("output/results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Method", "Avg Max Utilization"])
    writer.writerow(["ISP-only", ecmp_mlu])
    writer.writerow(["Random", random_mlu])
    writer.writerow(["GNN + PPO (Ours)", GNN_MLU])

print("Saved results to output/results.csv")
