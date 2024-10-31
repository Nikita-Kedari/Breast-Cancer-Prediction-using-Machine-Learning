# Breast-Cancer-Prediction-using-Machine-Learning
This project leverages machine learning algorithms to predict whether a breast tumor is **benign** or **malignant** using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset. The goal is to compare various models and identify the most accurate and reliable one for diagnosis.

## Overview
Breast cancer is a major health challenge, and early detection is crucial for improving patient outcomes. In this project, we implement and evaluate four machine learning algorithms:

- **Feedforward Neural Network (FNN)**  
- **Support Vector Machine (SVM)**  
- **Extreme Gradient Boosting (XGBoost)**  
- **Logistic Regression (LR)**  

Each model's performance is evaluated using key metrics: **accuracy, precision, recall, sensitivity, specificity,** and **AUC** (Area Under the Curve). 

---

## Dataset
**Name:** Wisconsin Diagnostic Breast Cancer (WDBC)  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  

- **Instances:** 569 samples  
- **Features:** 30 numeric tumor characteristics (e.g., radius, perimeter, concavity)  
- **Target Variable:**  
  - Benign (B) → 0  
  - Malignant (M) → 1  

The dataset is split into **80% training** and **20% testing** for model evaluation. 

---

## Models and Techniques
1. **Feedforward Neural Network (FNN):**  
   Captures non-linear patterns through hidden layers and backpropagation.
   
2. **Support Vector Machine (SVM):**  
   Trains on high-dimensional data with various kernels (linear, RBF).

3. **XGBoost:**  
   A fast, efficient gradient boosting algorithm ideal for structured data.

4. **Logistic Regression (LR):**  
   Serves as a baseline model with straightforward interpretability.

---

## Setup Instructions
1. **Install Required Libraries:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn tensorflow keras xgboost
   ```

2. **Clone or Download the Repository:**
   ```bash
   git clone <your-repository-url>
   cd <your-repository-folder>
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook breast_cancer_final.ipynb
   ```

---

## How to Run the Models
1. **Preprocess the Data:**
   - Missing values are handled.
   - The target labels are mapped: 'M' → 1, 'B' → 0.
   - Features are scaled using **StandardScaler**.

2. **Train and Evaluate the Models:**
   Execute the cells to train each model:
   ```python
   evaluate_model(fnn_model, "Feedforward Neural Network", X_train, y_train, X_test, y_test)
   evaluate_model(best_svm_model, "SVM", X_train, y_train, X_test, y_test)
   evaluate_model(best_xgb_model, "XGBoost", X_train, y_train, X_test, y_test)
   evaluate_model(logistic_model, "Logistic Regression", X_train, y_train, X_test, y_test)
   ```

3. **Performance Metrics:**
   Each model’s **accuracy, precision, recall, F1-score, sensitivity, specificity,** and **AUC** will be printed for both the training and testing datasets.

---

## Results Summary
| **Model**                    | **Accuracy (Test)** | **Precision** | **Recall** | **AUC** |
|------------------------------|--------------------|--------------|-----------|---------|
| Feedforward Neural Network   | 98.25%             | 100.00%      | 95.34%    | 99.14%  |
| Support Vector Machine (SVM) | 98.25%             | 100.00%      | 95.34%    | 97.67%  |
| XGBoost                      | 97.36%             | 97.61%       | 95.34%    | 96.97%  |
| Logistic Regression          | 97.36%             | 97.61%       | 95.34%    | 96.97%  |

- **FNN** and **SVM** achieved the highest test accuracy (98.25%), making them excellent candidates for breast cancer prediction.
- **XGBoost** and **Logistic Regression** also showed strong performance, proving the value of both ensemble models and simple linear classifiers.

---

## Future Improvements
- **Hyperparameter tuning:** Further refinement to optimize model performance.
- **Advanced Ensemble Models:** Explore stacking multiple algorithms.
- **Model Explainability:** Integrate SHAP or LIME for better interpretability.
- **Deployment:** Deploy the models in a web-based interface for real-time diagnosis.
- **Scalability:** Apply the models to larger, more complex datasets to test generalizability.

---

## Contributors
- **Sanika Mulik**  
- **Nikita Kedari**  
- **Juee Shinde**  

This project was made for TY BTech CSE: Artificial Intelligence and Expert Systems, in Semester V, July-Dec 2024

**Supervisor:** Prof. Pramod Mali  
**Institution:** MIT World Peace University, Pune
