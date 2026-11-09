# EMIR Data Card

Generated from `EMIR_Dataset_V1` with `scripts/run_eda.py`.

## Summary
- Total clips: **600**
- Sample rates observed: [16000, 44100]
- Channels observed: [1]

## Per-class statistics

| Class | Count | Total dur | Mean dur | Std | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Tizita | 162 | 81:00.00 | 30.00s | 0.00s | 30.00s | 30.00s |
| Bati | 144 | 72:00.00 | 30.00s | 0.00s | 30.00s | 30.00s |
| Ambassel | 147 | 73:30.00 | 30.00s | 0.00s | 30.00s | 30.00s |
| Anchihoye | 147 | 73:30.00 | 30.00s | 0.00s | 30.00s | 30.00s |

## RMS amplitude

| Class | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| Tizita | 0.2239 | 0.1153 | 0.0747 | 0.7305 |
| Bati | 0.1798 | 0.0652 | 0.0452 | 0.3198 |
| Ambassel | 0.1960 | 0.0805 | 0.0526 | 0.7333 |
| Anchihoye | 0.1869 | 0.0838 | 0.0342 | 0.4586 |

## Notes
- Source: Retta et al. 2023 (EMIR v1.0).
- Split: recording-level stratified 70/10/20 partition in `results/split_paper.json`.