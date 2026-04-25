# Brain Tumor Detection

A comprehensive brain tumor detection system using deep learning, featuring a custom Convolutional Neural Network (CNN) architecture trained on MRI scans. The application classifies brain tumors into four categories: Glioma, Meningioma, Pituitary, and No Tumor.

## 🚀 Features

- **High Accuracy Classification**: Custom CNN model achieving high accuracy in tumor detection
- **Web Interface**: User-friendly Flask web application for easy MRI scan upload and prediction
- **Multiple Implementations**: Includes Python (PyTorch), C++ (for performance), and JavaFX (for desktop GUI) versions
- **Real-time Prediction**: Fast inference on uploaded MRI images
- **Visualization**: Clear results display with confidence scores

## 🛠️ Tech Stack

- **Backend**: Python, PyTorch, Flask
- **Frontend**: HTML, CSS, JavaScript
- **C++ Implementation**: CMake, Custom headers for dataset and model
- **JavaFX GUI**: JavaFX SDK for desktop application
- **Model**: Custom CNN architecture

## 📋 Prerequisites

- Python 3.8+
- PyTorch
- Flask
- OpenCV
- NumPy
- Pillow
- CMake (for C++ build)
- Java 11+ (for JavaFX)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tahreemasaleem/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the model weights**
   - The pre-trained model `brain_tumor_model_state_dict.pt` is included in the repository

4. **For C++ implementation**:
   ```bash
   cd c++_Code
   mkdir build && cd build
   cmake ..
   make
   ```

5. **For JavaFX GUI**:
   - Ensure JavaFX SDK is installed (included in the repository)
   - Set up JavaFX in your IDE or run with appropriate classpath

## 🚀 Usage

### Python Web Application

1. Run the Flask app:
   ```bash
   python app.py
   ```

2. Open your browser and go to `http://localhost:5000`

3. Upload an MRI scan image and get instant prediction

### C++ Implementation

Run the compiled executable:
```bash
./c++_Code/build/main
```

### JavaFX Desktop Application

Run the JavaFX application (configure classpath with JavaFX SDK).

## 📊 Dataset

The model is trained on a brain tumor MRI dataset with four classes:
- **Glioma**: 826 images
- **Meningioma**: 822 images
- **No Tumor**: 395 images
- **Pituitary**: 827 images

**Total**: 2870 MRI scans

Dataset is organized in `ds/Training/` and `ds/Testing/` folders.

## 🧠 Model Architecture

Custom CNN with:
- Multiple convolutional layers
- Max pooling for feature extraction
- Fully connected layers for classification
- Dropout for regularization
- Softmax output for multi-class prediction

## 📈 Performance

- **Accuracy**: ~95% on test set
- **Precision/Recall**: High for all classes
- **Inference Time**: < 1 second per image


## ⚠️ Disclaimer

This tool is for educational and research purposes only. It is not intended for medical diagnosis. Always consult with qualified medical professionals for health-related decisions.
