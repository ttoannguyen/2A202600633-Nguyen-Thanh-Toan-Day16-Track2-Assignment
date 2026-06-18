#!/usr/bin/env python3
"""
LightGBM benchmark on Credit Card Fraud Detection (mlg-ulb/creditcardfraud).
Phương án CPU dự phòng cho Lab 16 — chạy trên r5.2xlarge.

Sinh đủ 10 metrics của bảng 7.6 và ghi ra benchmark_result.json (deliverable 7.8).

Cách chạy (trên CPU node, trong ~/ml-benchmark có creditcard.csv):
    python3 benchmark.py
"""
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
)

CSV_PATH = "creditcard.csv"
RANDOM_STATE = 42

# 1. Load data --------------------------------------------------------------
t0 = time.perf_counter()
df = pd.read_csv(CSV_PATH)
load_time = time.perf_counter() - t0
print(f"Loaded {len(df):,} rows in {load_time:.3f}s")

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# scale_pos_weight để bù mất cân bằng (fraud ~0.17%)
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos

# 2. Train ------------------------------------------------------------------
train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "scale_pos_weight": scale_pos_weight,
    "n_jobs": -1,
    "verbosity": -1,
    "seed": RANDOM_STATE,
}

t0 = time.perf_counter()
model = lgb.train(
    params,
    train_set,
    num_boost_round=1000,
    valid_sets=[valid_set],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)
train_time = time.perf_counter() - t0
best_iteration = model.best_iteration
print(f"Trained in {train_time:.3f}s, best_iteration={best_iteration}")

# 3. Evaluate ---------------------------------------------------------------
y_proba = model.predict(X_test, num_iteration=best_iteration)
y_pred = (y_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_proba)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# 4. Inference latency / throughput ----------------------------------------
one_row = X_test.iloc[[0]]
N = 100
t0 = time.perf_counter()
for _ in range(N):
    model.predict(one_row, num_iteration=best_iteration)
latency_1row_ms = (time.perf_counter() - t0) / N * 1000

batch = X_test.iloc[:1000]
t0 = time.perf_counter()
model.predict(batch, num_iteration=best_iteration)
batch_time = time.perf_counter() - t0
throughput_1000 = 1000 / batch_time

# 5. Report -----------------------------------------------------------------
results = {
    "load_time_sec": round(load_time, 3),
    "train_time_sec": round(train_time, 3),
    "best_iteration": int(best_iteration),
    "auc_roc": round(float(auc), 4),
    "accuracy": round(float(acc), 4),
    "f1_score": round(float(f1), 4),
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "inference_latency_1row_ms": round(latency_1row_ms, 3),
    "inference_throughput_1000rows_per_sec": round(throughput_1000, 1),
    "instance_type": "r5.2xlarge",
    "dataset": "mlg-ulb/creditcardfraud",
    "n_rows": int(len(df)),
}

print("\n===== BENCHMARK RESULT =====")
for k, v in results.items():
    print(f"{k:40s}: {v}")

with open("benchmark_result.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nWrote benchmark_result.json")
