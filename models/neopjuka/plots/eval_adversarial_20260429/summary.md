# Wakeword Evaluation Dashboard

- model: `/Users/unripeplum/projects/nubjuk/wakeword/models/neopjuka/train/20260429_011002/data/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`
- stride: `1` (forced:1 (training_stride=3 from /Users/unripeplum/projects/nubjuk/wakeword/models/neopjuka/train/20260429_011002/data/training_config.yaml))
- positives: `30` clips
- negatives: `775` clips
- negative hours: `0.4077` h
- ROC AUC (clip-level): `0.6351`
- PR AUC (clip-level): `0.4782`
- cutoff `0.78` => FRR `0.0667`, FAPH `2168.0837`
- recommended threshold (target FAPH <= 0.50): `1.00`

## Files
- `learning_curves.png`
- `score_distribution.png`
- `roc_pr_curve.png`
- `threshold_tradeoff.png`
- `threshold_metrics.csv`
- `score_table.csv`
