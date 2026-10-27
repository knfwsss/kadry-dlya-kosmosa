---
license: cc-by-4.0
task_categories:
- time-series-forecasting
tags:
- time-series
- forecasting
- anomaly-detection
- classification
- TSLib
- tsfile
- modality:timeseries
- timeseries
- format:tsfile
pretty_name: SMAP (TsFile)
size_categories:
- 1M<n<10M
configs:
- config_name: default
  data_files:
  - split: train
    path: SMAP_train.tsfile
  - split: test
    path: SMAP_test.tsfile
modality:
- timeseries
- tabular
- text
language:
- en
---

# SMAP (TsFile)

Apache TsFile version of the `SMAP` anomaly-detection subset of
[`thuml/Time-Series-Library`](https://huggingface.co/datasets/thuml/Time-Series-Library).

## Overview

Soil Moisture Active Passive satellite telemetry (25 channels) for anomaly detection.

- **Train:** 135,183 rows (all normal).
- **Test:** 427,617 rows (with per-timestep 0/1 anomaly labels).
- **Channels:** 25.

The train and test segments are stored as two separate TsFiles
(`SMAP_train.tsfile` / `SMAP_test.tsfile`), preserving the original split.

## Schema (TsFile structure)

- **Time** (INT64, milliseconds) — row index * 1000 ms (the `.npy` source has no timestamp; a monotone 1 Hz proxy axis).
- **FIELD** (25 channels, FLOAT) — the sensor/metric channels.
- **label** (INT32) — per-timestep anomaly flag (0/1). The train file is all 0
  (no ground-truth labels); the test file carries the anomaly labels.

No channels or rows are dropped.

## Usage

Read the `.tsfile` files with the Apache TsFile Java or Python SDK.

## Source & license

- Original dataset: https://huggingface.co/datasets/thuml/Time-Series-Library (subset `SMAP`)
- Author / publisher: thuml (Tsinghua University)
- Paper: https://arxiv.org/abs/2407.13278
- License: CC BY 4.0
