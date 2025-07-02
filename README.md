# 📊 Simple Linear Regression Portfolio Project

"""
🎯 Objective:
Build and evaluate a simple linear regression model using a complete pipeline:
1. Data loading and inspection
2. Cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Model training and visualization
5. Performance evaluation with standard and advanced metrics
6. Final equation for inference

📁 Dataset:
Salary Dataset - Simple Linear Regression
Used in the "Machine Learning A - Z" course (Kaggle dataset)

💡 Technologies used: Python, Pandas, Scikit-learn, Seaborn, Matplotlib
"""

# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, explained_variance_score,
    mean_squared_log_error, max_error
)
from scipy import stats

# === 1. Data Collection ===
from google.colab import files
uploaded = files.upload()
df = pd.read_csv(next(iter(uploaded)))
display(df.head())

# === 2. Data Overview ===
print("\n📋 DataFrame Info")
df.info()
print("\n📊 Descriptive Statistics")
print(df.describe(include='all'))

# Rename numeric columns to x and y for simplicity
expected_cols = ['x', 'y']
if not all(col in df.columns for col in expected_cols):
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if len(num_cols) >= 2:
        df = df.rename(columns={num_cols[0]: 'x', num_cols[1]: 'y'})
        print(f"🔁 Renamed '{num_cols[0]}' to 'x' and '{num_cols[1]}' to 'y'")

# === 3. Data Cleaning ===
df.replace("", np.nan, inplace=True)
df.dropna(subset=['x', 'y'], inplace=True)
df = df[(df['x'] != 0) & (df['y'] != 0)]
z_scores = np.abs(stats.zscore(df[['x', 'y']]))
df = df[(z_scores < 3).all(axis=1)]

# === 4. EDA ===
print("\nMissing Values:")
print(df.isnull().sum())
df.hist(bins=30, figsize=(10, 4), color='skyblue')
plt.suptitle('Variable Distributions')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['x'])
plt.title('Boxplot of x')
plt.subplot(1, 2, 2)
sns.boxplot(x=df['y'])
plt.title('Boxplot of y')
plt.tight_layout()
plt.show()

sns.scatterplot(x='x', y='y', data=df)
plt.title('Scatter Plot: x vs y')
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# === 5. Model Training ===
X = df[['x']]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
pipeline.fit(X_train, y_train)
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# === 6. Regression Equation ===
coef = pipeline.named_steps['regressor'].coef_[0]
intercept = pipeline.named_steps['regressor'].intercept_
print(f"\n📐 Regression Line: y = {coef:.4f} * x + {intercept:.4f}")

# === 7. Plot Fit Line ===
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, y_pred_train, color='navy', label='Train Fit Line')
plt.scatter(X_test, y_test, color='green', marker='x', s=80, label='Test data')
plt.plot(X_test, y_pred_test, color='lime', label='Test Predictions')
plt.title("Linear Regression - Prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# === 8. Evaluation Metrics ===
print("\n✅ Model Evaluation Metrics")
print("MAE:", mean_absolute_error(y_test, y_pred_test))
print("MSE:", mean_squared_error(y_test, y_pred_test))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("R²:", r2_score(y_test, y_pred_test))

def adjusted_r2(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

print("Adjusted R²:", adjusted_r2(r2_score(y_test, y_pred_test), X_test.shape[0], X_test.shape[1]))
print("Median AE:", median_absolute_error(y_test, y_pred_test))
print("Explained Variance:", explained_variance_score(y_test, y_pred_test))
print("Max Error:", max_error(y_test, y_pred_test))

# === 9. Advanced Metrics ===
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

print("MAPE:", mean_absolute_percentage_error(y_test, y_pred_test))
print("SMAPE:", symmetric_mape(y_test, y_pred_test))

if (y_test > 0).all() and (y_pred_test > 0).all():
    print("MSLE:", mean_squared_log_error(y_test, y_pred_test))
else:
    print("MSLE: skipped (requires strictly positive values)")

# === 10. Residuals Plot ===
residuals = y_test - y_pred_test
plt.figure(figsize=(8, 5))
plt.scatter(y_pred_test, residuals, color='purple')
plt.axhline(y=0, color='gray', linestyle='--')
plt.title("Residuals vs Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.grid(True)
plt.show()

# === Final Inference Instruction ===
print("\nℹ️ Predict new values using: y = {:.4f} * x + {:.4f}".format(coef, intercept))


