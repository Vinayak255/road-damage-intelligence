# Model Selection

## Why Object Detection for Road Damage?

### Computer Vision Task Comparison

| Approach | Output | Localization | Suitable? |
|----------|--------|--------------|-----------|
| **Image Classification** | Single label per image | No | ❌ Cannot locate individual damages |
| **Object Detection** | Bounding boxes + class + confidence | Yes | ✅ Ideal for locating multiple damages |
| **Semantic Segmentation** | Pixel-level mask | Yes (precise) | ⚠️ Overkill, slower, harder to train |
| **Instance Segmentation** | Per-object pixel mask | Yes (precise) | ⚠️ More accurate but higher complexity |

**Object detection** is the most appropriate task because:
1. We need to **locate** each damage instance (not just classify the whole image)
2. Multiple damages can exist in a single image
3. Bounding boxes provide sufficient localization for damage assessment
4. It offers a good balance between accuracy and inference speed

### Key Object Detection Concepts

#### Bounding Boxes
Rectangular regions defined by (x1, y1, x2, y2) coordinates that enclose detected objects. Each detection produces one bounding box.

#### Confidence Score
A value between 0.0 and 1.0 representing the model's certainty that the detected region contains the predicted class. Higher scores indicate more confident predictions.

#### Intersection over Union (IoU)
Measures overlap between predicted and ground-truth bounding boxes:

```
IoU = Area of Overlap / Area of Union
```

- IoU = 1.0: Perfect match
- IoU = 0.5: Commonly used threshold for "correct" detection
- IoU = 0.0: No overlap

#### Non-Maximum Suppression (NMS)
When multiple overlapping detections exist for the same object, NMS keeps only the highest-confidence detection and removes redundant ones. Controlled by the IoU threshold parameter.

#### Speed/Accuracy Trade-off
Larger models (YOLOv8x) are more accurate but slower. Smaller models (YOLOv8n) are faster but less accurate. The choice depends on the deployment scenario.

### Why YOLO?

**YOLO (You Only Look Once)** is selected for this project because:

| Factor | YOLO Advantage |
|--------|----------------|
| **Speed** | Single-pass detection enables real-time inference |
| **Accuracy** | Competitive with two-stage detectors for this task |
| **Ease of use** | Ultralytics provides excellent training/inference API |
| **Community** | Large community, well-documented, many examples |
| **Tracking** | Built-in ByteTrack integration for video |
| **Transfer learning** | Easy to fine-tune on custom datasets like RDD2022 |

**YOLO is NOT claimed to be universally the best model.** For certain applications (medical imaging, satellite imagery), other architectures may be more appropriate. YOLO is chosen here because its speed-accuracy trade-off is well-suited for road damage detection where near-real-time processing is desirable.

### YOLOv8 Architecture Overview

YOLOv8 uses:
- **CSPDarknet backbone**: Feature extraction from input images
- **PANet neck**: Multi-scale feature aggregation
- **Decoupled head**: Separate classification and regression heads
- **Anchor-free design**: Direct bounding box prediction

### Model Variants

| Model | Parameters | mAP (COCO) | Speed | Use Case |
|-------|-----------|------------|-------|----------|
| YOLOv8n | 3.2M | 37.3 | Fastest | Edge/mobile deployment |
| YOLOv8s | 11.2M | 44.9 | Fast | Balanced |
| YOLOv8m | 25.9M | 50.2 | Medium | Good accuracy |
| YOLOv8l | 43.7M | 52.9 | Slow | High accuracy |
| YOLOv8x | 68.2M | 53.9 | Slowest | Maximum accuracy |

For road damage detection, **YOLOv8s or YOLOv8m** offer a reasonable balance.

### Evaluation Metrics

- **Precision**: Of all detections made, what fraction are correct?
- **Recall**: Of all actual damages, what fraction did we detect?
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean AP averaged across IoU thresholds from 0.5 to 0.95
