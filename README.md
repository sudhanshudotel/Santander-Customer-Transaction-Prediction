# Santander-Customer-Transaction-Prediction

## Project Overview

This repository documents a comprehensive machine learning workflow designed for the Santander Customer Transaction Prediction competition. The challenge involves predicting whether customers will make a transaction based on their historical data. Our project encompasses the entire process from initial setup in Google Colab, through data preprocessing, model training, and hyperparameter tuning, as detailed below.

### Problem Statement
The goal of the Santander Customer Transaction Prediction competition is to identify which customers will make a transaction in the future. This binary classification problem involves working with a highly anonymized dataset where feature names and values are masked, adding complexity to the modeling process.

### Environment Configuration
- **Google Colab:** Utilized for executing the project with access to Google Drive for dataset and script storage.
- **Warning Management:** Configured to ignore non-critical warnings for clearer output.
- **Visualization:** Matplotlib settings were customized to enhance the clarity and readability of visualizations.
- **TensorFlow 2.x:** Enabled to take advantage of its latest features for potential deep learning implementations.
- **Reproducibility:** Ensured by setting a fixed random seed in TensorFlow and NumPy.

### Data Preprocessing
- **Dataset Handling:** Training and testing data were loaded from Google Drive, with copies made to preserve the integrity of the original datasets.
- **Data Splitting:** Segregated training data into training and validation datasets (80-20 split) for effective model performance validation.
- **Feature Consistency:** Uncommon features across data splits were identified and removed to ensure consistency.
- **Identifier Management:** Non-predictive identifiers were removed from the datasets.
- **Missing Data Handling:** Combined datasets to uniformly address missing values, applying the most frequent strategy for imputation.
- **Data Encoding:** Implemented one-hot encoding for categorical features and label encoding for the target variable.

### Feature Engineering
- **Normalization:** Utilized MinMaxScaler to ensure that all model inputs have a uniform scale.

### Model Setup
- **Model Selection:** Configured Logistic Regression and MLP Classifier models to address class imbalance and facilitate convergence.
- **Pipeline Integration:** Each model was encapsulated within a pipeline to streamline processing and validation.

### Hyperparameter Tuning
- **GridSearchCV:** Systematically explored hyperparameters using predefined splits for validation.
- **Parameter Grid Exploration:** Comprehensive grids for Logistic Regression and MLP Classifier were defined and explored.
- **Performance Analysis:** The best parameters and scores were captured and documented. Detailed results were stored in structured CSV files for analysis.

### Model Performance

The hyperparameter tuning process yielded the following results for our models:

- **MLP Classifier:**
  - **Best Score:** 0.674081
  - **Parameters:** 
    - Alpha: 1e-06
    - Learning Rate Initialization: [value from best_param]
  - **Configuration:** Early stopping enabled to prevent overfitting.

- **Logistic Regression:**
  - **Best Score:** 0.640026
  - **Parameters:** 
    - C: 0.12
    - Tolerance: 1e-06
  - **Configuration:** Class weights balanced to address potential class imbalance.

These scores were obtained using the `f1_macro` scoring metric, which considers both precision and recall, making it a robust measure in scenarios with class imbalance.

