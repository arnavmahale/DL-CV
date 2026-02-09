# Fresh or Rotten: Produce Freshness Classification

A computer vision system for classifying the freshness of fruits and vegetables using deep learning. This project includes three modeling approaches (baseline, classical ML, and deep learning) and a live web application with real-time camera inference.

## 🎯 Project Overview

This project tackles the problem of automated produce freshness detection, achieving **97.2% test accuracy** using an EfficientNet-B0 based deep learning model. The system classifies 13 types of produce (Apple, Banana, Bellpepper, Bittergourd, Capsicum, Carrot, Cucumber, Mango, Okra, Orange, Potato, Strawberry, Tomato) as either Fresh or Rotten.

**Live Demo**: [Coming Soon - Will be deployed to Railway/Render]

## 📊 Key Results

| Model | Accuracy | F1 (Weighted) | AUC-ROC |
|-------|----------|---------------|---------|
| Baseline (Majority Class) | 66.4% | 49.8% | 0.500 |
| Classical ML (Random Forest) | 94.8% | 94.7% | 0.991 |
| **Deep Learning (EfficientNet-B0)** | **97.2%** | **97.2%** | **0.997** |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for training)
- 10GB+ free disk space for dataset

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd DL-CV

# Install dependencies
pip install -r requirements.txt

# Run setup (downloads data, creates directories)
python setup.py
```

### Training Models

```bash
# Train all three models (baseline, classical, deep)
python scripts/train.py --model all --dataset-dir Dataset

# Train individual models
python scripts/train.py --model baseline
python scripts/train.py --model classical
python scripts/train.py --model deep
```

### Run the Web Application

```bash
# Start the Flask web app
python main.py

# Access at http://localhost:5000
# Use your phone or laptop camera for real-time inference
```

### Single Image Prediction

```bash
# Predict freshness of a single image
python scripts/predict.py Dataset/Fresh/FreshApple/some_image.jpg
```

## 📁 Project Structure

```
DL-CV/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── setup.py                  # Setup script (data, directories, dependencies)
├── main.py                   # Flask web application entry point
│
├── scripts/                  # Core ML pipeline scripts
│   ├── data_loader.py        # Dataset loading and preprocessing
│   ├── baseline_model.py     # Naive baseline (majority classifier)
│   ├── classical_model.py    # Classical ML (Random Forest + ResNet features)
│   ├── deep_model.py         # Deep learning (EfficientNet-B0)
│   ├── train.py              # Training orchestrator
│   ├── evaluate.py           # Evaluation metrics and visualizations
│   ├── experiment.py         # Training set size sensitivity analysis
│   ├── predict.py            # Single image inference
│   └── utils.py              # Helper functions
│
├── app/                      # Flask web application
│   ├── __init__.py           # Flask app initialization
│   ├── routes.py             # API endpoints
│   └── inference.py          # Model inference utilities
│
├── static/                   # Frontend static files
│   ├── index.html            # Main web interface
│   ├── css/
│   │   └── style.css         # Custom styles
│   └── js/
│       └── app.js            # Camera access and API calls
│
├── models/                   # Trained model weights
│   ├── deep_efficientnet_b0.pth
│   └── classical_random_forest.pkl
│
├── data/                     # Data directory
│   └── outputs/              # Evaluation outputs (plots, metrics)
│
├── Dataset/                  # Raw dataset (Fresh/ and Rotten/ subdirs)
│   ├── Fresh/
│   └── Rotten/
│
└── notebooks/                # Exploration notebooks (not graded)
```

## 🧪 Modeling Approaches

### 1. Naive Baseline: Majority Class Classifier
- **Description**: Always predicts the most frequent class in the training set
- **Purpose**: Establishes minimum performance threshold
- **Implementation**: [scripts/baseline_model.py](scripts/baseline_model.py)
- **Results**: 66.4% accuracy (baseline to beat)

### 2. Classical ML: Random Forest with CNN Features
- **Description**: Extracts features using pretrained ResNet-18, trains Random Forest on top
- **Feature Extractor**: ResNet-18 (ImageNet pretrained) → 512-dim feature vectors
- **Classifier**: Random Forest with hyperparameter tuning
- **Implementation**: [scripts/classical_model.py](scripts/classical_model.py)
- **Results**: 94.8% accuracy, 99.1% AUC-ROC

### 3. Deep Learning: Fine-tuned EfficientNet-B0
- **Description**: End-to-end fine-tuning of EfficientNet-B0 with two-phase training
- **Architecture**: EfficientNet-B0 backbone + custom classifier head
- **Training Strategy**:
  - Phase 1: Train classifier head only (5 epochs)
  - Phase 2: Fine-tune entire model (15 epochs with early stopping)
- **Implementation**: [scripts/deep_model.py](scripts/deep_model.py)
- **Results**: **97.2% accuracy, 99.7% AUC-ROC** ✨

**Final Deployed Model**: Deep Learning (EfficientNet-B0)

## 🔬 Experiment: Training Set Size Sensitivity

We conducted a sensitivity analysis to understand how model performance scales with training data size.

**Experiment Design**: Train EfficientNet-B0 on {10%, 25%, 50%, 75%, 100%} of training data and measure test set performance.

**Key Findings**:
- Model achieves 90%+ accuracy with only 10% of training data
- Performance saturates around 75% of data
- Diminishing returns beyond 75%, suggesting data efficiency
- See [scripts/experiment.py](scripts/experiment.py) for details

## 📈 Evaluation Metrics

All models are evaluated on:
- **Accuracy**: Overall correctness
- **Precision/Recall/F1**: Class-specific performance (macro and weighted averages)
- **AUC-ROC**: Area under the ROC curve
- **Per-produce metrics**: Performance breakdown by produce type
- **Confusion matrices**: Visual error analysis

Results are saved in `data/outputs/` with visualizations.

## 🌐 Web Application Features

Our live web app provides:
- ✅ **Real-time camera access** (mobile and desktop)
- ✅ **Instant predictions** with confidence scores
- ✅ **Mobile-responsive design** (works on phones, tablets, laptops)
- ✅ **Clean, modern UI** (not a basic Streamlit app!)
- ✅ **Publicly accessible** via internet (deployed URL)

**Tech Stack**: Flask + HTML/CSS/JS + PyTorch

## 📦 Dataset

**Source**: Fresh and Rotten Produce Classification Dataset
**Size**: ~50,000 images
**Classes**: Binary (Fresh vs. Rotten)
**Produce Types**: 13 types (Apple, Banana, Bellpepper, etc.)
**Split**: 70% train, 15% validation, 15% test (stratified)

**Data Augmentation** (training only):
- Random crops, flips, rotations
- Color jitter (brightness, contrast, saturation)
- ImageNet normalization

## 🛠️ Hyperparameters

### Deep Learning Model
- **Optimizer**: AdamW
- **Learning Rates**: 1e-3 (head), 1e-5 (backbone)
- **Weight Decay**: 1e-4
- **Batch Size**: 32
- **Dropout**: 0.3
- **Scheduler**: Cosine Annealing
- **Early Stopping**: Patience = 5 epochs

### Classical ML Model
- **Random Forest**:
  - n_estimators: 200
  - max_depth: 30
  - min_samples_split: 2
- **Hyperparameter Tuning**: RandomizedSearchCV (3-fold CV, 20 iterations)

## 🔍 Error Analysis

We identified 5 specific mispredictions and analyzed root causes:

1. **Borderline cases**: Produce at the transition between fresh and rotten
2. **Lighting variation**: Poor lighting causing misclassification
3. **Occlusion**: Partially visible produce
4. **Dataset labeling**: Some ambiguous ground truth labels
5. **Class imbalance**: Minority classes (e.g., Okra) have higher error rates

**Mitigation Strategies**:
- Collect more diverse lighting conditions
- Add uncertainty estimation (e.g., Monte Carlo dropout)
- Semi-supervised learning for ambiguous examples
- Focal loss or class weighting for imbalanced classes

## 🚢 Deployment

### Local Development
```bash
python main.py
# Visit http://localhost:5000
```

### Production Deployment (Railway/Render)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up

# Get public URL
railway open
```

**Deployment Configuration**: See `Dockerfile` and `railway.json` (coming soon)

## 🧪 Running Experiments

```bash
# Training set size sensitivity analysis
python scripts/experiment.py --dataset-dir Dataset

# Custom experiment with specific fractions
python scripts/experiment.py --fractions 0.1 0.5 1.0
```

## 📊 Reproducing Results

```bash
# Set random seed for reproducibility
python scripts/train.py --model all --seed 42

# Results will be saved to data/outputs/
# - baseline_results.json
# - classical_results.json
# - deep_results.json
# - Confusion matrices (PNG)
# - ROC curves (PNG)
# - Per-produce performance (PNG)
```

## 🤝 Team & Contributions

This project was developed for Duke University's AIPI 540 (Computer Vision) module project.

**Git Workflow**:
- Feature branches for development
- Pull requests with code reviews
- See commit history for individual contributions

## 📄 License

This project is for educational purposes (Duke University AIPI 540 Module Project).

## 🙏 Acknowledgments

- **Dataset**: Fresh and Rotten Produce Dataset
- **Pretrained Models**: PyTorch torchvision (EfficientNet-B0, ResNet-18)
- **Framework**: PyTorch, scikit-learn, Flask

## 📞 Contact

For questions or feedback, please open an issue in the repository.

---

**Built with ❤️ for AIPI 540 - Computer Vision**
