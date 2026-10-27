# 🏎️ Used Car Valuation ML

## Table of Contents

* [Project Overview](#-project-overview)
* [Interactive Web App](#-interactive-web-app)
* [Project Structure](#-project-structure)
* [Dataset](#-dataset)
* [Workflow](#-workflow)
* [Technologies and Libraries](#-technologies-and-libraries)
* [Installation and Usage](#-installation-and-usage)
* [Key Results](#-key-results)
* [Contributions](#-contribution)
* [Acknowledgements](#-acknowledgements)

## 📋 Project Overview

A comprehensive machine learning project for predicting used car prices using various regression models with hyperparameter tuning. This project implements a complete machine learning pipeline for predicting used car prices, including data preprocessing, exploratory data analysis, hyperparameter optimization using Optuna and Streamlit application hosting on Render.

## 🚀 Interactive Web App
**[View the Live Web App on Render](https://used-car-valuation-ml.onrender.com)**

## 📂 Project Structure

```
├── app.py                          # Streamlit web application
├── best_params.json                # Optimal hyperparameters from Optuna
├── cat_dict.joblib                 # Mapping dictionary for categorical data
├── eda.ipynb                       # Exploratory data analysis (EDA)
├── custom_transformer.py           # Custom Scikit-Learn transformer (OrderedTargetEncoder)
├── final_pipeline.joblib           # Serialized production-ready model
├── data_preprocessing.ipynb        # Data cleaning and feature engineering
├── model_tuning.ipynb              # Model training and optimization (Google Colab)
├── requirements.txt                # Production dependencies (Render)
├── requirements-dev.txt            # Development dependencies (Notebooks)
├── used_cars.csv                   # Raw dataset from Kaggle
└── used_cars_preprocessed.csv      # Cleaned and processed data
```

## 📊 Dataset

**Raw dataset**

**Source:** [Used Cars Dataset - Kaggle](https://www.kaggle.com/datasets/leiwangetc666/used-cars)

**Size:** 4009 rows 12 features

**Preprocessed dataset**

**Size:** 3998 rows 11 features

## 🔄 Workflow

### 1. Data Preprocessing ([`data_preprocessing.ipynb`](data_preprocessing.ipynb))
- Handles data cleaning
- Feature Engineering new columns
- Outlier detection and treatment
- Output: [`used_cars_preprocessed.csv`](used_cars_preprocessed.csv)

### 2. Exploratory Data Analysis ([`eda.ipynb`](eda.ipynb))
- Univariate and multivariate analysis
- Distribution analysis of numerical data
- Correlation analysis with target variable
- Statistical validation, including Variance Inflation Factor (VIF) analysis to detect and mitigate multicollinearity.

### 3. Custom Scikit-Learn Transformer ([`custom_transformer.py`](custom_transformer.py))
- A dedicated module featuring the OrderedTargetEncoder. This custom transformer manages high-cardinality categorical features with their smoothed target means, and then maps these smoothed means to an ordered integer sequence.

### 4. Model Development & Hyperparameter Tuning ([`model_tuning.ipynb`](model_tuning.ipynb))[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rv8kAe6bXNKtF5NGXjmYdEfBZNvKbJkr?usp=sharing)
This notebook performs extensive hyperparameter optimization using Optuna.
**⚠️ Note:** This notebook was executed in **Google Colab** due to intensive computational requirements for hyperparameter optimization.

#### Running in Google Colab
Since the original notebook uses Google Drive, follow these steps to use `model_tuning.ipynb` in Google Colab:

1. **Upload files to Google Drive:**
   ```
   /MyDrive/folder_path/
   ├── model_tuning.ipynb
   └── used_cars_preprocessed.csv
   ```

2. **In the notebook, mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Set working directory:**
   ```python
   import os
   os.chdir('/content/drive/MyDrive/folder_path')
   ```

**Models Implemented:**
- Linear and Regualarized Linear Models
- Ridge & Lasso Regression
- K-Nearest Neighbors (KNN), Support Vector Regression
- Regression Trees (Decision Tree, Random Forest)
- Boosting Regressor
- XGBoost
- Ensemble Methods

**Optimization Framework:** Optuna with Bayesian optimization (TPE Sampler) with implementation of pruning

### 5. Production Model and Hyperparameter Results

- [`best_params.json`](best_params.json): Stores the best hyperparameter settings found for each model, ensuring reproducibility and easy reference for tuning 
- [`final_pipeline.joblib`](final_pipeline.joblib): The serialized, complete machine learning pipeline featuring all transformations and the final, best-performing XGB Regressor model, optimized for deployment.

### 6. Web Application ([`app.py`](app.py))
- A user-friendly Streamlit interface that allows users to input car specifications (brand, year, mileage, etc.).

- Integrated with the production pipeline to deliver real-time price predictions.

- Features dynamic input handling, mapping unknown user inputs to an "Other" category to ensure model stability.

### 7. Deployment & Environment Configuration
- [`requirements.txt`](requirements.txt): A dependency file for Render deployment. It contains only the essential libraries needed to run the web application on Render.

- [`requirements-dev.txt`](requirements-dev.txt): A comprehensive dependency list for local development, including Jupyter, pandas, Matplotlib, and Seaborn for data science workflows.

- [`cat_dict.joblib`](cat_dict.joblib): A serialized dictionary containing the unique categorical values. This is used by the app to validate user inputs.


## 📦 Technologies and Libraries

| Category | Tools |
| :--- | :--- |
| **Data Manipulation** | ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) |
| **Machine Learning** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-eb8a2f?style=for-the-badge&logo=xgboost&logoColor=white) |
| **Statistical Analysis** | ![Statsmodels](https://img.shields.io/badge/statsmodels-%232d5e80.svg?style=for-the-badge&logo=statsmodels&logoColor=white) |
| **Optimization & Serialization** | ![Optuna](https://img.shields.io/badge/Optuna-blue?style=for-the-badge) ![Joblib](https://img.shields.io/badge/Joblib-orange?style=for-the-badge) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![Seaborn](https://img.shields.io/badge/Seaborn-blue?style=for-the-badge) |
| **Application Framework** | ![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white) |
| **Hosting** | ![Render](https://img.shields.io/badge/Render-%46E3B7.svg?style=for-the-badge&logo=render&logoColor=white)|

### Key Library Functions:
- **Pandas & NumPy**: Essential for data cleaning, transformation, and handling numerical arrays.
- **Scikit-learn**: Used for building the model pipeline, cross-validation, and implementing diverse algorithms.
- **Statsmodels**: Specifically utilized for advanced statistical testing, including **Variance Inflation Factor (VIF)** analysis to ensure feature independence.
- **Optuna**: Powers the automated hyperparameter tuning process for maximum model and time efficiency.
- **XGBoost**: High-performance gradient boosting with integration of GPU.
- **Matplotlib & Seaborn**: Used to generate exploratory data analysis (EDA) plots and feature correlation heatmaps.
- **Joblib**: Used for saving and loading the finalized `final_pipeline.joblib` for production-ready deployment.
- **Streamlit** : Used to transform the machine learning pipeline into an interactive web dashboard for real-time user predictions.

## 💾 Installation and Usage

### For Production (App only)
```bash
pip install -r requirements.txt
streamlit run app.py
```
### For Development (Notebooks + App)
```bash
pip install -r requirements-dev.txt
jupyter notebook
```

## 📈 Key Results
### Best Model - XGB Regressor
### Best Hyperparameters
```json
"best_params": {
   "smoothing": 0.16685680111960388,
   "booster": "gbtree",
   "eta": 0.5576453681503339,
   "gamma": 0,
   "max_depth": 4,
   "min_child_weight": 10,
   "subsample": 0.9203980828433391,
   "colsample_bytree": 0.6542646785426075,
   "learning_rate": 0.19398370999714648,
   "n_estimators": 457,
   "reg_alpha": 0.0002128343321412847,
   "reg_lambda": 0.042607318592429354
}
```
### Model Performance
- **Test R² Score:** 0.869
- **Mean Absolute Error:** 0.210* 
- **Mean Squared Error:** 0.086*
*Calculated on log-transformed scale

## 🤝 Contribution

Contributions are welcome to enhance model accuracy or add new features.

- Fork the Project.
- Create your Feature Branch (`git checkout -b feature-name`).
- Commit your Changes (`git commit -m 'Add new feature'`).
- Push to the Branch (`git push origin feature-name`).
- Open a Pull Request.

## 🙏 Acknowledgements

This project was built using the Python ecosystem and various open-source tools. We leverage Jupyter Notebooks and Google Colab for interactive development and GPU-accelerated training. Core data processing is handled by NumPy and Pandas, with visualizations powered by Matplotlib and Seaborn. The machine learning pipeline is built on Scikit-Learn and XGBoost, featuring automated hyperparameter tuning via Optuna and efficient serialization with Joblib. The final application is built using Streamlit and deployed on Render, using datasets sourced from Kaggle.
