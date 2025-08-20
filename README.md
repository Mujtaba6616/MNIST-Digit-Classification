# 🖥️ MNIST Digit Classification

This project implements a **handwritten digit classifier** using the MNIST dataset. The model is built with **Keras (TensorFlow backend)** and uses a feed-forward **neural network** to recognize digits (0–9) from grayscale images.

## 📌 Project Overview

* **Objective:** Classify handwritten digits (28x28 grayscale images) into 10 categories (0–9).
* **Dataset:** MNIST — 60,000 training images & 10,000 test images.
* **Frameworks/Libraries:** Keras, TensorFlow, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn.
* **Output:** Trained model with evaluation metrics (accuracy, loss curves, confusion matrix).

## ⚙️ Installation & Requirements

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd MNIST_Digit_Classification
pip install -r requirements.txt
```

**Requirements:**
* Python 3.7+
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

## 📂 Project Structure

```
MNIST_Digit_Classification.ipynb   # Jupyter Notebook (end-to-end workflow)
README.md                          # Project documentation
requirements.txt                   # Python dependencies
```

## 🚀 Workflow

1. **Data Preparation**
   * Load MNIST dataset from Keras.
   * Normalize pixel values (0–255 → 0–1).
   * Convert labels to one-hot encoded vectors.

2. **Model Architecture**
   * **Input Layer:** 784 neurons (flattened 28x28 image).
   * **Hidden Layers:** Dense + Dropout for regularization.
   * **Output Layer:** 10 neurons (Softmax activation).

3. **Training**
   * **Loss Function:** Categorical Crossentropy.
   * **Optimizer:** Adam.
   * **Metric:** Accuracy.
   * Trained on 60,000 samples with a validation split.

4. **Evaluation**
   * Test set accuracy.
   * Confusion matrix for misclassification analysis.
   * Error visualization for misclassified digits.

## 📊 Results

* Achieved ~97% accuracy on the test dataset.
* Confusion matrix highlights digit-wise performance.
* Misclassified samples show similarity between certain digits (e.g., 4 vs 9).

## 📈 Example Outputs

* **Confusion Matrix:** Visual representation of prediction accuracy across classes.
* **Misclassified Digits:** Examples where the model failed to predict correctly.

## 🔮 Future Improvements

* Implement Convolutional Neural Networks (CNNs) for higher accuracy.
* Use data augmentation to improve generalization.
* Perform hyperparameter tuning (batch size, learning rate, dropout).

## 🧑‍💻 Author

**Mujtaba Ahmed**  
Senior Computer Science Student — FAST-NUCES Lahore
