# Wakeword Evaluation Dashboard

- model: `/Users/unripeplum/projects/nubjuk/wakeword/models/neopjuka/train/20260429_011002/data/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`
- stride: `1` (forced:1 (training_stride=3 from /Users/unripeplum/projects/nubjuk/wakeword/models/neopjuka/train/20260429_011002/data/training_config.yaml))
- positives: `30` clips
- negatives: `10` clips
- negative hours: `0.0086` h
- ROC AUC (clip-level): `0.9367`
- PR AUC (clip-level): `0.9836`
- cutoff `0.78` => FRR `0.0667`, FAPH `582.5490`
- recommended threshold (target FAPH <= 0.50): `0.85`

## Files
- `learning_curves.png`
- `score_distribution.png`
- `roc_pr_curve.png`
- `threshold_tradeoff.png`
- `threshold_metrics.csv`
- `score_table.csv`
