# Wakeword Evaluation Dashboard

- model: `/Users/unripeplum/projects/nubjuk/wakeword/models/neopjuka/release/wake_nubjuk_ko.tflite`
- stride: `1` (default:1)
- positives: `20` clips
- negatives: `10` clips
- negative hours: `0.0086` h
- ROC AUC (clip-level): `0.8400`
- PR AUC (clip-level): `0.9317`
- cutoff `0.78` => FRR `0.1500`, FAPH `466.0392`
- recommended threshold (target FAPH <= 0.50): `1.00`

## Files
- `learning_curves.png`
- `score_distribution.png`
- `roc_pr_curve.png`
- `threshold_tradeoff.png`
- `threshold_metrics.csv`
- `score_table.csv`
