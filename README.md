# Road Damage Intelligence System

![Road Damage Detection](https://img.shields.io/badge/Computer_Vision-YOLOv8-blue.svg)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-lightgrey.svg)
![VITyarthi](https://img.shields.io/badge/Project-VITyarthi-orange.svg)

A complete Computer Vision application that automatically detects, classifies, and estimates the severity of road damage in images and videos. Developed for the VITyarthi "Build Your Own Project" university evaluation.

## 🌟 Features

- **Image & Video Analysis**: Upload images or videos for automated inspection.
- **YOLOv8 Object Detection**: High-performance detection of four road damage classes (RDD2022 dataset).
- **Damage Classification**:
  - `D00`: Longitudinal Crack
  - `D10`: Transverse Crack
  - `D20`: Alligator Crack
  - `D40`: Pothole
- **Object Tracking**: Uses ByteTrack to deduplicate damage events in video streams.
- **Severity Estimation**: Heuristic visual engine to categorize severity as Low, Medium, or High.
- **Evidence Generation**: Automatically crops and saves annotated evidence of each detected damage.
- **Analytics Dashboard**: Real-time Plotly charts based on database records.
- **History & Reporting**: SQLite persistence of all analysis sessions and detections.
- **Modular Architecture**: Clean separation of CV, API, Database, and Business Logic.

## 🛠️ Technologies / Tools Used
- **Language**: Python 3.10+
- **Deep Learning**: Ultralytics YOLOv8, PyTorch
- **Computer Vision**: OpenCV, ByteTrack
- **Backend Framework**: FastAPI, Uvicorn
- **Database**: SQLite, SQLAlchemy
- **Frontend**: HTML5, CSS3, JavaScript, Jinja2
- **Analytics & Visualization**: Plotly
- **Testing**: Pytest
## 🏗️ Architecture & Documentation

The project is heavily documented to explain all design decisions and components. Please review the following documents in the `docs/` folder:

- [Problem Statement](statement.md)
- [System Architecture](docs/architecture.md)
- [Model Selection](docs/model_selection.md)
- [Design Decisions](docs/design_decisions.md)
- [Dataset Documentation](docs/dataset.md)
- [Evaluation Documentation](docs/evaluation.md)

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- A modern web browser

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Copy the example environment file and configure it:

```bash
# On Windows
copy .env.example .env

# On Linux/Mac
cp .env.example .env
```

*Note: The application will run without a trained model, but image/video analysis will return errors until one is provided.*

### 4. Provide a Trained Model (Crucial)

This application is built for **inference**. It requires a trained YOLO model to perform detection.
If you don't have one, you can train one using the provided scripts (see below).

Place your trained model file at: `models/best.pt`

### 5. Start the Application

```bash
python scripts/run_demo.py
```
Open your browser and navigate to: `http://localhost:8000`

## 🛠️ Developer Scripts

The repository includes several utility scripts in the `scripts/` directory:

- **Initialize Database**: `python scripts/setup_db.py`
- **Run Server Demo**: `python scripts/run_demo.py`
- **Prepare Dataset**: `python scripts/prepare_dataset.py --source /path/to/RDD2022`
- **Train Model**: `python scripts/train.py --data datasets/rdd2022/data.yaml`
- **Evaluate Model**: `python scripts/evaluate_model.py`

## 🧪 Testing

The project includes a comprehensive pytest suite covering all modules:

```bash
pytest
```

To run with coverage:

```bash
pytest --cov=app --cov-report=term-missing
```

## 📁 Project Structure

```
road-damage-intelligence/
├── app/                  # Application Source Code
│   ├── api/              # FastAPI routes and schemas
│   ├── cv/               # OpenCV & YOLO computer vision pipeline
│   ├── analysis/         # Severity engine and statistics logic
│   ├── database/         # SQLAlchemy ORM and repository
│   ├── config/           # Centralized settings
│   ├── evaluation/       # Model evaluation wrapper
│   └── utils/            # Logging, validators, exceptions
├── frontend/             # Web UI
│   ├── templates/        # Jinja2 HTML templates
│   └── static/           # CSS, JS, Fonts
├── scripts/              # CLI scripts for DB, Training, Eval
├── tests/                # Pytest suite
├── docs/                 # Project documentation
├── data/                 # Auto-generated (uploads, output, evidence)
├── models/               # Store best.pt here
└── requirements.txt      # Python dependencies
```

```

## 📸 Screenshots
   <img width="1332" height="958" alt="Screenshot 2026-09-18 114716" src="https://github.com/user-attachments/assets/abf8a302-ed42-4053-a6b3-782fc07d9aca" />
  <img width="1315" height="898" alt="Screenshot 2026-09-18 114728" src="https://github.com/user-attachments/assets/16560bb8-f625-4e36-ab00-b6a43d042c2a" />
  <img width="1307" height="473" alt="Screenshot 2026-09-18 114735" src="https://github.com/user-attachments/assets/cf955e9f-fdca-4f2b-8298-e4b8280556cf" />





##Demo Video
https://drive.google.com/file/d/1_L_tG02_7kU-PJoV77IHhP0GEGp57yOJ/view?usp=sharing
NAME: Vinayak Bhadauria
Reg No.:24BAI10240
