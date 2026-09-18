# Dataset Documentation

## RDD2022 — Road Damage Dataset 2022

### Overview

The **Road Damage Dataset 2022 (RDD2022)** is a large-scale road damage detection dataset used in the Crowdsensing-based Road Damage Detection Challenge (CRDDC'2022). It contains road surface images captured from multiple countries with bounding box annotations for various types of road damage.

### Official Source

- **Paper**: "RDD2022: A Multi-National Image Dataset for Automatic Road Damage Detection" (IEEE Big Data 2022)
- **Challenge**: CRDDC'2022 — IEEE Big Data Cup Challenge
- **Download**: [https://github.com/sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector)
- **Alternative**: [https://figshare.com/articles/dataset/RDD2022/21431547](https://figshare.com/articles/dataset/RDD2022/21431547)

### Damage Classes (CRDDC2022)

This project uses the four standard CRDDC2022 damage categories:

| Class Code | Name | Description |
|-----------|------|-------------|
| D00 | Longitudinal Crack | Cracks running parallel to the road direction |
| D10 | Transverse Crack | Cracks running perpendicular to the road direction |
| D20 | Alligator Crack | Network of interconnected cracks resembling alligator skin |
| D40 | Pothole | Cavity/depression in the road surface |

### Dataset Structure

The RDD2022 dataset is organized by country:

```
RDD2022/
├── Japan/
│   └── train/
│       ├── images/
│       │   ├── Japan_000001.jpg
│       │   ├── Japan_000002.jpg
│       │   └── ...
│       └── annotations/
│           └── xmls/
│               ├── Japan_000001.xml
│               ├── Japan_000002.xml
│               └── ...
├── India/
│   └── train/
│       ├── images/
│       └── annotations/
│           └── xmls/
├── Czech/
├── Norway/
├── United_States/
└── China_MotorBike/
```

### Annotation Format

Annotations are in **Pascal VOC XML format**:

```xml
<annotation>
    <size>
        <width>600</width>
        <height>600</height>
    </size>
    <object>
        <name>D40</name>
        <bndbox>
            <xmin>120</xmin>
            <ymin>200</ymin>
            <xmax>350</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
</annotation>
```

### How to Download

1. Visit the [RDD2022 GitHub repository](https://github.com/sekilab/RoadDamageDetector)
2. Download the dataset for desired countries
3. Extract to a local directory
4. Use the preparation script to convert to YOLO format:

```bash
python scripts/prepare_dataset.py --source path/to/RDD2022 --output datasets/rdd2022
```

### Dataset Preparation

The `prepare_dataset.py` script:
- Validates image-annotation pairs
- Converts Pascal VOC XML to YOLO format
- Creates train/validation splits (default 80/20)
- Generates `data.yaml` for Ultralytics training

### Expected Output Structure

After preparation:

```
datasets/rdd2022/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

### Important Notes

- The full RDD2022 dataset is **NOT included** in this repository
- The repository does **NOT** automatically download the dataset
- The model has **NOT** been pre-trained on RDD2022 unless explicitly done by the user
- Dataset bias exists: image characteristics vary by country (camera, road type, lighting)
