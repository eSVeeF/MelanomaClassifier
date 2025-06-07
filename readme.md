# MelanomaClassifier

A deep learning project for classifying skin lesions as melanoma or non-melanoma using medical images. This repository implements a pipeline that combines Convolutional Neural Networks (CNNs) and Multi-Layer Perceptrons (MLPs), leveraging the PH2Dataset for training and validation.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f209710d-a7fb-40e4-9674-3ebf4fd4a30e" alt="image" width="400"/>
</p>

## 🧠 Features

- Modular CNN and MLP models for flexible experimentation
- Custom cosine learning rate scheduler for smoother convergence
- 5-fold and 10-fold cross-validation for robust performance evaluation
- Dataset loading and preprocessing tailored to the PH2 dataset
- Support for training and inference workflows

## 📂 Project Structure

| File/Folder | Purpose |
|-------------|---------|
| `PreProcessing/` | Image and data preprocessing tools |
| `PH2Dataset/` | Contains the dataset (if not excluded) |
| `FeatureBuilders/` | Scripts to extract additional features |
| `results/` | Accuracies and recalls of all models |
| `1D_10fold_CNN.py` | 10-fold CV implementation of the 1-dimension CNN|
| `CNNClassifier.py` | CNN model architecture |
| `CNNClassifier5FoldCV.py` | 5-fold CV for CNN |
| `CustomLearningRateScheduler.py` | Learning rate strategy |
| `MLPClassifier.py` | MLP model for tabular features |
| `MLPClassifierGridSearch.py` | MLP model with Grid Search |
| `mod_PH2_dataset.csv` | Processed metadata |
| `read_images.py` | Image reading and preprocessing |

---

## 📦 Installation

```bash
git clone https://github.com/eSVeeF/MelanomaClassifier.git
cd MelanomaClassifier
pip install -r requirements.txt
```

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
#### MLP Classifier 
To run the Multilayer Perceptron (MLP) classifier with grid search:
```bash
python MLPClassifierGridSearch.py.py
```
#### 1D 10-Fold CNN
To execute the 1-dimensional CNN 10-fold:
```bash
python 1D_10fold_CNN.py
```
#### CNN Classifier
To train the Convolutional Neural Network (CNN) with 5-fold cross-validation:
```bash
python CNNClassifier5FoldCV.py
```

## 📉 Sample Results
The models achieve a competitive **82%** accuracy and **91%** recall on PH2 data using standard metrics. Code is modular and ready for adaptation to other medical imaging datasets.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or feature requests.

## License
This project is licensed under the MIT License, feel free to use and modify for non-commercial purposes.
