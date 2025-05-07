# Step 1: Import necessary libraries and load the dataset
import pandas as pd

# Load the dataset
file_path = 'diabetes.csv'
df = pd.read_csv(file_path)

# Display first 5 rows of the dataframe
df.head()

# Step 2: Handle missing values and normalize data

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Replace zeros with NaN in selected columns
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_columns] = df[zero_columns].replace(0, np.nan)

# Fill missing values with mean of each column
df[zero_columns] = df[zero_columns].fillna(df.mean())

# Normalize the data using Min-Max Scaler (scale to range [0,1])
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Separate features and target
X = df_scaled.drop('Outcome', axis=1)
y = df['Outcome']

# Display first 5 rows after preprocessing
df_scaled.head()

# Step 3: Split the data into training and testing sets

from sklearn.model_selection import train_test_split

# Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Print shapes of the resulting datasets
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

# Step 4: Train the Logistic Regression model

from sklearn.linear_model import LogisticRegression

# Create the Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Print model intercept and coefficients (optional)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Step 5.1: Evaluate model using Accuracy

from sklearn.metrics import accuracy_score

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Step 5.2: Confusion Matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display it visually
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Also print the matrix values
print("Confusion Matrix Values:")
print(cm)

# Step 5.3: Calculate Precision, Recall, F1-Score

from sklearn.metrics import classification_report

# Generate the classification report
report = classification_report(y_test, y_pred)
print(report)

# Step 5.4: Calculate ROC-AUC Score

from sklearn.metrics import roc_auc_score

# Get predicted probabilities for class 1 (diabetic)
y_proba = model.predict_proba(X_test)[:, 1]

# Compute ROC-AUC score
roc_auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC Score:", roc_auc)


# Step 5.4: Plot ROC Curve

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Get predicted probabilities for class 1 (diabetic)
y_proba = model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Compute AUC score
roc_auc = roc_auc_score(y_test, y_proba)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random classifier line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()



