# Project Report: Diabetes Prediction using Logistic Regression

## 🎯 Objective:
The goal of this project is to classify individuals into two categories: "Diabetic" and "Non-Diabetic" using the **Logistic Regression** algorithm. We used the well-known **Pima Indians Diabetes Dataset** for this task.

---

## 📊 Dataset:
- **Source**: Pima Indians Diabetes Dataset
- **Number of Rows**: 768
- **Number of Columns**: 9
- **Features**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (Label: 0 = Non-Diabetic, 1 = Diabetic)

---

## 🛠️ Data Preprocessing:
1. **Handling Missing Values**:
   - Zero values in some columns (e.g., `Glucose`, `BloodPressure`) were treated as missing data.
   - Missing values were replaced with the mean of their respective columns.

2. **Normalization**:
   - All features were normalized to the range `[0,1]` using `MinMaxScaler`.

---

## ⚙️ Train/Test Split:
- **Split Ratio**: 80% Training / 20% Testing
- Used `random_state=42` for reproducibility.
- **Training Data Shape**: `(614, 8)`
- **Testing Data Shape**: `(154, 8)`

---

## 📈 Model Training:
- The `LogisticRegression()` model from the `scikit-learn` library was used.
- The model was trained on the training dataset.
- Coefficients (`coef_`) and intercept (`intercept_`) were computed.

---

## 📊 Evaluation Results:

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.7792 |
| **ROC-AUC Score** | 0.8226 |
| **Precision (Class 1)** | 0.73 |
| **Recall (Class 1)** | 0.60 |
| **F1-Score (Class 1)** | 0.66 |

---

## 🧮 Confusion Matrix:

|                | Predicted 0 | Predicted 1 |
|----------------|-------------|-------------|
| **Actual 0**   | TN = 87     | FP = 12     |
| **Actual 1**   | FN = 22     | TP = 33     |

- **TN (True Negative)**: 87 individuals correctly classified as non-diabetic.
- **FP (False Positive)**: 12 individuals incorrectly classified as diabetic.
- **FN (False Negative)**: 22 individuals incorrectly classified as non-diabetic.
- **TP (True Positive)**: 33 individuals correctly classified as diabetic.

---

## 📉 Strengths and Weaknesses:

### ✅ Strengths:
- **Good Accuracy (77.9%)**
- **High AUC (0.82)** → Indicates good ability to separate the two classes.
- **Reasonable Precision (73%)** → Among those predicted as diabetic, 73% were actually diabetic.

### ❌ Weaknesses:
- **Low Recall (60%)** → The model only identified 60% of the actual diabetic individuals.
- **High False Negatives (22)** → The model missed diagnosing many diabetic individuals, which could be critical.

---

## 📈 ROC Curve:
The ROC Curve shows that our model performs **well** in separating the two classes.  
With an AUC of 0.82, the model outperforms random guessing.

---

## 💡 Conclusion:
The Logistic Regression model performed reasonably well in predicting diabetes but has room for improvement:
- We can use techniques like **SMOTE** or **Class Weighting** to address class imbalance.
- We can experiment with other models like **Random Forest**, **XGBoost**, or **Neural Networks**.
- We can perform **Hyperparameter Tuning** to optimize model performance.