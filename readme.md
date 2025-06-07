# MelanomaClassifier

A deep learning project for classifying skin lesions as melanoma or non-melanoma using medical images. This repository implements a pipeline that combines Convolutional Neural Networks (CNNs) and Multi-Layer Perceptrons (MLPs), leveraging the PH2Dataset for training and validation. Report.pdf contains the full story

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
The model is trained on the PH2 Dataset and organized in the following format:
```bash
MelanomaClassifier/
├── PH2Dataset/
│   └── Custom Images/
│       ├── Lesion/
│       ├── Normal/
│       └── Others/
```

## 🚀 Usage
### MLP Classifier 
To run the Multilayer Perceptron (MLP) classifier
```bash
python MLPClassifier.py
```
### 1D 10-Fold CNN
To execute the 1-dimensional CNN
```bash
python 1D_10fold_CNN.py
```
### CNN Classifier
To run the Convolutional Neural Network (CNN) classifier
```bash
python CNNClassifier.py
```

## 🔧 Configuration
Model parameters, optimizer settings, and data paths can be configured through command-line arguments or by editing the train.py script.

## 📊 Evaluation
Performance metrics such as accuracy, recall, and loss are computed and logged during cross-validation. Plots and logs are saved in the results/ directory.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or feature requests.

## License
This project is licensed under the MIT License. See LICENSE for details.
