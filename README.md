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

1. **Data Cleaning**:

   - Removed rows with null/NaN values, then 765 rows were remaining.
   - Dropped columns of 'id' and 'school_year' as intended to make a general model.
   - Few rows were with 0.0 BMI, set their who_bmi to 'Normal', as its the mode for complete dataset.

2. **Exploratory Data Analysis** :

   - **Understanding Data:** I used a summary function to see the distribution of age, depression scores, and anxiety scores.

   - **Gender and Depressiveness:** Females in the dataset tend to report feeling more depressive than males.

   - **BMI and Depressiveness:** As BMI category increases (from normal to overweight to obese), the proportion of people reporting depressiveness also tends to increase. However, most of the people in the dataset fall into the "normal" and "overweight" BMI categories, so these groups have the largest number of people reporting depressiveness simply because there are more people in those categories.

   - **BMI and Depressiveness (Males):** Among males, the same pattern holds – higher BMI categories are associated with more depressiveness, with a particularly high level among obese males.

   - **BMI and Depressiveness (Females):** The same trend is seen in females, but the proportion of depressive females is generally higher than males across all BMI categories.

   - **Age and Depressiveness:** Most people reporting depressiveness are between 18 and 24 years old, with the highest number at age 19.

   - **Age and Depressiveness (Correlation):** There's a negative relationship between age and depressiveness, meaning that as age increases, reported depressiveness tends to decrease (though this doesn't imply causation).

   - **Age and Anxiousness (Correlation):** A similar negative relationship exists between age and anxiousness.

   - **Age of Depressed Individuals:** The average age of people reporting depressiveness is around 20, with most being in their late teens and early twenties.

   - **Age and Anxiousness Pattern:** The relationship between age and anxiousness is similar to that of age and depressiveness.

   - **Age and Sleepiness:** The relationship between age and sleepiness is different and less clear compared to age and depressiveness/anxiousness. There are fewer people reporting sleepiness, and they are spread across a wider age range.

   - **Depression Diagnosis:** A small proportion of people who report feeling depressive have actually received a formal diagnosis of depression.

   - **Depression Treatment:** Even fewer people who report depressiveness are receiving formal treatment for it.

   - **Suicidal Thoughts and Depressiveness:** People who report suicidal thoughts almost always report feeling depressive.

   - **Anxiety Diagnosis and Treatment:** Similar to depression, few people who report feeling anxious have received a formal diagnosis or treatment for anxiety.

   - **Sleepiness with Anxiety and Depression:** About half of the people who report feeling anxious or depressive also report feeling sleepy.

   - **Suicidal Thoughts and Anxiousness:** People who report suicidal thoughts may or may not also report feeling anxious.

   - **Depression Severity by Gender:** Most males report mild or minimal/no depression severity, while females report mild, minimal/none, and moderate severity.

   - **Anxiety Severity by Gender:** Most males report minimal/none followed by mild anxiety, whereas females mostly report mild and then minimal/none.

   - **Depression Severity by Age:** Across different age groups, the order of depression severity from most to least common is generally: mild > minimal/none > moderate > moderately severe > severe > none.

   - **Anxiety Severity by Age:** A similar pattern is observed for anxiety severity across age groups: mild > minimal/none > moderate > severe.

   **Correlations**

   - **Suicidal vs. Depressiveness** (0.50):Moderate Positive Correlation: This suggests a moderately strong tendency for individuals with higher levels of depressiveness to also report higher levels of suicidal ideation or attempts.
   - **Suicidal vs. Anxiousness** (0.27):Weak Positive Correlation: There's a slight tendency for individuals with higher levels of anxiousness to also report more suicidal thoughts, but the relationship is not as strong.
   - **Suicidal vs. Sleepiness** (0.17):Very Weak Positive Correlation: The relationship between sleepiness and suicidal thoughts is minimal. There might be a slight tendency, but it's not a strong or reliable association.
   - **Depressiveness vs. Anxiousness** (0.47): Moderate Positive Correlation: Individuals with higher levels of depressiveness tend to also report higher levels of anxiousness, indicating that these two often co-occur.
   - **Depressiveness vs. Sleepiness** (0.23): Weak Positive Correlation: Similar to the relationship with suicidal thoughts, there is a weak association between depressiveness and sleepiness.
   - **Anxiousness vs. Sleepiness** (0.23): Weak Positive Correlation: A weak association exists between anxiousness and sleepiness.

3. **Encoding**:

   - **Boolean Features**: The boolean features ['depressiveness', 'suicidal', 'depression_diagnosis', 'depression_treatment', 'anxiousness', 'anxiety_diagnosis', 'anxiety_treatment', 'sleepiness'] were converted to integer type, with True/False (or similar) values represented as 1 and 0, respectively.
   - **Ordinal Categorical Features**: Ordinal categorical features ( 'who_bmi', 'depression_severity', and 'anxiety_severity') were transformed into numerical sequences using a mapping. This mapping preserved the order of the categories (e.g., 'Mild' might be mapped to 1, 'Moderate' to 2, etc.). The specific mapping used should be documented elsewhere.
   - **Gender**: The 'gender' feature was encoded using 0 and 1 to represent the two distinct gender categories. The specific assignment of 0 and 1 to each gender should be documented.

   **New Correlations Found**

   **Strong Correlations (Positive or Negative):**

   - **phq_score (Depression Score) and gad_score (Anxiety Score) (0.66):** A strong positive correlation suggests individuals with higher depression scores tend to have higher anxiety scores.
   - **depression_severity and gad_score (0.63):** Similar to the above, higher depression severity is associated with higher anxiety scores.
   - **gad_score and anxiety_severity (0.95):** This very high correlation is expected as anxiety_severity is likely derived from gad_score.
   - **anxiety_severity and anxiousness (0.79):** A high correlation is expected given the similarity of these measures.
   - **depression_diagnosis and depression_treatment (0.69):** Individuals diagnosed with depression are more likely to receive treatment.
   - **anxiety_diagnosis and anxiety_treatment (0.69):** Similarly, individuals diagnosed with anxiety are more likely to receive treatment.

   **Moderate Correlations:**

   - **depressiveness and suicidal (0.50):** Reinforces the link between depressiveness and suicidal ideation.
   - **phq_score and depressiveness (0.74):** High correlation suggests self-reported depressiveness aligns with phq_score.
   - **gad_score and depressiveness (0.50):** Higher anxiety scores are associated with higher levels of reported depressiveness.
   - **anxiousness and depressiveness (0.47):** Suggests a moderate overlap between anxiety and depressive symptoms.

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
