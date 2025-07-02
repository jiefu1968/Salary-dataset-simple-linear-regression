# 📊 Simple Linear Regression Portfolio Project

## 🎯 Objective

Build and evaluate a simple linear regression model using a complete machine learning pipeline.

## 📁 Dataset

- **Name:** Salary Dataset - Simple Linear Regression  
- **Source:** [Kaggle - Machine Learning A-Z Course](https://www.kaggle.com/datasets)

## 💡 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Google Colab

---

## 🧭 Project Pipeline

### 1. Data Loading
- Upload CSV directly in Colab using `files.upload()`.

### 2. Data Overview
- Print shape, types, descriptive statistics, and null values.

### 3. Data Cleaning
- Remove NaNs, zeroes, and outliers via Z-score.

### 4. EDA (Exploratory Data Analysis)
- Histograms, boxplots, scatter plots, and correlation matrix.

### 5. Model Training
- Pipeline: Imputation → Scaling → Linear Regression.

### 6. Regression Equation
- Display the regression line in the form: `y = a * x + b`.

### 7. Visualizations
- Fit line over training/test data.
- Residuals vs predicted values.

### 8. Model Evaluation

**Basic Metrics:**
- MAE, MSE, RMSE
- R² and Adjusted R²
- Median AE, Max Error

**Advanced Metrics:**
- MAPE, SMAPE
- MSLE (when applicable)
- Explained Variance Score

---

## 📌 Instructions

You can predict new outputs using the printed regression equation:
```python
y = a * x + b

## 📌 Results (Example)

📊 Descriptive Statistics
       Unnamed: 0  YearsExperience         Salary
count   30.000000        30.000000      30.000000
mean    14.500000         5.413333   76004.000000
std      8.803408         2.837888   27414.429785
min      0.000000         1.200000   37732.000000
25%      7.250000         3.300000   56721.750000
50%     14.500000         4.800000   65238.000000
75%     21.750000         7.800000  100545.750000
max     29.000000        10.600000  122392.000000

✅ Model Evaluation Metrics
MAE: 0.43114132445431047
MSE: 0.29537212911990357
RMSE: 0.5434814892155054
R²: 0.9522158960665235
Adjusted R²: 0.9402698700831543
Median AE: 0.4553690344062167
Explained Variance: 0.9534275402944385
Max Error: 0.8085460599334056
MAPE: 7.527598029053885
SMAPE: 7.1671371989522115
MSLE: 0.006224245518426202

Regression Line: y = 2.7357 * x + 5.4391

