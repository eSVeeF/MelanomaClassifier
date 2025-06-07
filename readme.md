# MelanomaClassifier

A deep learning project for classifying skin lesions as melanoma or non-melanoma using medical images. This repository implements a pipeline that combines Convolutional Neural Networks (CNNs) and Multi-Layer Perceptrons (MLPs), leveraging the PH2Dataset for training and validation.

![image](https://github.com/user-attachments/assets/f209710d-a7fb-40e4-9674-3ebf4fd4a30e)


## 📂 Project Structure

- `CNNClassifier.py` — Defines the CNN architecture for image-based classification.
- `MLPClassifier.py` — Implements an MLP model for auxiliary features.
- `CombinedModel.py` — Combines CNN and MLP outputs into a unified classifier.
- `CustomLRScheduler.py` — Contains a cosine annealing learning rate scheduler.
- `PH2Dataset.py` — Loads and preprocesses the PH2 skin lesion dataset.
- `train.py` — Main training script using cross-validation.
- `inference.py` — Inference script to run predictions on new data.
- `utils.py` — Utility functions for metric calculation, plotting, etc.

## 🧠 Features

- Modular CNN and MLP models for flexible experimentation.
- Custom cosine learning rate scheduler for smoother convergence.
- K-fold cross-validation for robust performance evaluation.
- Dataset loading and preprocessing tailored to the PH2 dataset.
- Support for training and inference workflows.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/eSVeeF/MelanomaClassifier.git
   cd MelanomaClassifier
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

## 🖼️ Dataset
The model is trained on the PH2 Dataset. Please download it from the official source (due to licensing restrictions) and organize it in the following format:
```bash
MelanomaClassifier/
├── data/
│   └── PH2Dataset/
│       ├── images/
│       └── masks/
```

## 🚀 Usage
### Training
```bash
python train.py --epochs 50 --batch_size 32 --lr 0.001 --folds 5
```
### Inference
```bash
python inference.py --image_path ./sample_image.jpg --model_path ./models/best_model.pth
```

## 🔧 Configuration
Model parameters, optimizer settings, and data paths can be configured through command-line arguments or by editing the train.py script.

## 📊 Evaluation
Performance metrics such as accuracy, F1-score, and confusion matrices are computed and logged during cross-validation. Plots and logs are saved in the results/ directory.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or feature requests.
