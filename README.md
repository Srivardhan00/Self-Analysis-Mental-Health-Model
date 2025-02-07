# Self-Analysis Mental Health Model

## Overview

This project is a machine learning-based tool for predicting mental health conditions using user-provided symptoms. It includes data preprocessing, multiple model comparisons, and an interactive interface for inference.

---

## About Dataset

- **File** :`depression_anxiety_data.csv`
- **Source** : `https://www.kaggle.com/datasets/shahzadahmad0402/depression-and-anxiety-data`
- **Samples**: 783
- **Features**: 19

### Dataset Features:

- id: unique key
- school_year: Current school year of the participant (ranges from 1 to 4). [Numerical (Ordinal)]
- age: Age of the participant. [Numerical (Continuous)
- gender: Gender of the participant. [Categorical (Nominal)]
- bmi: Body Mass Index value of the participant. [Numerical (Continuous)]
- who_bmi: Classification of WHO BMI based on the bmi value. [Categorical (Ordinal)]
- phq_score: Measure of the severity of symptoms related to depression (numerical score). [Numerical (Discrete)]
- depression_severity: Category of depression based on the phq_score (e.g., Minimal, Mild, Moderate, Moderately Severe, Severe). [Categorical (Ordinal)]
- depressiveness: A boolean feature, indicating the presence of depressiveness. [Boolean]
- suicidal: A boolean feature, indicating suicidal ideation or attempts. [Boolean]
- depression_diagnosis: A boolean feature, indicating a formal diagnosis of depression. [Boolean]
- depression_treatment: A boolean feature, indicating a formal treatment of depression. [Boolean]
- gad_score: Measure of the severity of symptoms related to generalized anxiety disorder (numerical score). [Numerical (Discrete)]
- anxiety_severity: Category of anxiety based on the gad_score (e.g., Minimal, Mild, Moderate, Severe). [Categorical (Ordinal)]
- anxiousness: A boolean feature, indicating the presence of anxiety. [Boolean]
- anxiety_diagnosis: A boolean feature, indicating a formal diagnosis of anxiety. [Boolean]
- anxiety_treatment: A boolean feature, indicating a formal treatment of anxiety. [Boolean]
- epworth_score: A score from the Epworth Sleepiness Scale, measuring daytime sleepiness. [Numerical (Discrete)]
- sleepiness: A boolean feature, indicating the presence of sleepiness. [Boolean]

### Target Variables

- depression_severity
- anxiety_severity

### **Limitation**

The data was collected from a sample of university students, so the findings may not be generalizable to other populations.

## Dataset Preprocessing Steps

1. **Data Cleaning**: Missing values are handled using imputation strategies.
2. **Feature Engineering**: Symptoms and questionnaire scores are transformed into numerical features.
3. **Encoding**: Labels are encoded into categorical values.
4. **Scaling**: Standardization is applied using `StandardScaler`.

---

## Exploratory Data Analysis (EDA)

Before training the models, EDA was performed to understand the dataset:

1. **Distribution Analysis**: Histograms and boxplots were used to visualize the distribution of PHQ and GAD scores.
2. **Correlation Analysis**: A heatmap was generated to identify relationships between features.
3. **Missing Value Analysis**: Checked for NaN values and handled them appropriately.
4. **Class Distribution**: Bar plots were used to observe class imbalances in depression and anxiety severity levels.
5. **Feature Importance**: Used feature importance scores from Random Forest to select the most relevant features.

---

## Feature Selection and Model Comparison

1. **Feature Selection**: After identifying important features, a combined dataset was created with selected variables for both depression and anxiety.
2. **Model Training**: Two models were compared using GridSearchCV for hyperparameter tuning:
   - **Logistic Regression**: Used as a baseline model.
   - **Random Forest**: Evaluated for better performance and robustness.
3. **Final Model Choice**: The best model was chosen based on cross-validation performance and evaluation metrics.

### **Final Model Selection:**

The models were evaluated based on the following metrics:

- **Classification Report**
- **Accuracy**
- **Precision & Recall**
- **F1-score**
- **ROC-AUC Score**

The final models are saved using `joblib` for inference.

---

## How to Run the Project

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **2. Running the Jupyter Notebook for EDA and Training**

```bash
jupyter notebook predict_mental_health.ipynb
```

### **3. Running the Streamlit Web App for Inference**

```bash
streamlit run mental_health_ui.py
```

---

## File Structure

```
|-- .gitignore                 # Ignores unnecessary files
|-- depression_anxiety_data.csv # Dataset used for training
|-- predict_mental_health.ipynb # Jupyter Notebook for model training & inference
|-- mental_health_ui.py         # Streamlit-based Web UI for mental health analysis
|-- requirements.txt            # Dependencies needed to run the project
|-- rf_depression.joblib        # Trained Random Forest model for depression
|-- rf_anxiety.joblib           # Trained Random Forest model for anxiety
|-- scaler.joblib               # StandardScaler used for preprocessing
```

---

## Contributors

- **Srivardhan Veeramalli**
