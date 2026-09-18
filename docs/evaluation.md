# Evaluation Documentation

## Model Evaluation

### Metrics

The model evaluation reports standard object detection metrics:

| Metric | Description |
|--------|-------------|
| **Precision** | Fraction of detected objects that are correct |
| **Recall** | Fraction of actual objects that were detected |
| **mAP@0.5** | Mean Average Precision at IoU threshold 0.5 |
| **mAP@0.5:0.95** | Mean AP averaged across IoU thresholds 0.5 to 0.95 (stricter) |

### Running Evaluation

```bash
# Using the default model path from .env
python scripts/evaluate_model.py

# With explicit arguments
python scripts/evaluate_model.py --model models/best.pt --data datasets/rdd2022/data.yaml --imgsz 640
```

### Output

Results are saved to:
- `reports/evaluation_results.json` — machine-readable metrics
- Terminal output — human-readable summary

### Viewing Results

- **Web UI**: Navigate to `/evaluation` in the frontend
- **API**: `GET /api/evaluation`

### Important Notes

- **Evaluation has NOT been pre-executed** — results only appear after you run the evaluation script
- Metrics are **never fabricated** — if evaluation hasn't been run, the system reports "Evaluation not yet executed"
- Requires a trained road-damage model and a validation dataset
- mAP@0.5:0.95 is the stricter metric and is typically lower than mAP@0.5
