# 🩺 Diabetes Prediction Project

## 📌 Project Overview

This project predicts whether a patient is diabetic (`Outcome = 1`) or not (`Outcome = 0`) using the **Pima Indians Diabetes Dataset**.
The goal is to practice the **end-to-end supervised machine learning pipeline**:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Preprocessing (scaling, handling missing values, outliers)
* Model Training (Logistic Regression, KNN, Decision Tree, Random Forest)
* Evaluation & Comparison

---

## 📂 Dataset

* **Source**: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* **Rows**: 768 patients
* **Features**: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, DiabetesPedigreeFunction, Age
* **Target**: `Outcome` → 0 (No Diabetes), 1 (Diabetes)

⚠️ Some features had invalid values (e.g., Glucose = 0, BMI = 0). These were treated as **missing values** and imputed.

---

## 🛠 Steps in the Project

### 1. Data Cleaning

* Replaced invalid `0` values in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI with **nan**.
* Handled missing values using **KNN Imputer**.

📌 *Reason*: Models cannot handle NaN values.

---

### 2. Exploratory Data Analysis (EDA)

* **Histograms** → Understand distributions and skewness.
* **Boxplots** → Detect outliers.
* **Correlation Heatmap** → Identify important features (Glucose had the strongest correlation with diabetes).
* **Class Balance Check** → Checked how many patients had diabetes vs not.

📌 *Reason*: EDA helps understand the dataset before modeling, and guides preprocessing/feature selection.

---

### 3. Preprocessing

* **RobustScaler** applied to features.

📌 *Reason*: Scaling ensures fair comparison between features. RobustScaler was chosen because it is **less sensitive to outliers** (important for medical data like Insulin or BMI).

---

### 4. Model Building

Trained and compared 4 supervised learning models:

* **Logistic Regression** → Linear, interpretable, good baseline.
* **K-Nearest Neighbors (KNN)** → Distance-based, sensitive to scaling.
* **Decision Tree** → Rule-based, interpretable, but prone to overfitting.
* **Random Forest** → Ensemble of trees, more robust than a single tree.

---

### 5. Evaluation

Metrics used:

* **Accuracy** → Overall correctness.
* **Precision (for diabetics)** → Of predicted diabetics, how many were correct?
* **Recall (for diabetics)** → Of actual diabetics, how many were detected? (⚠️ Very important in medical prediction).
* **F1-score** → Balance between precision & recall.
* **Confusion Matrix** → Showed exact True/False Positives and Negatives.

📌 *Reason*: Accuracy alone is not enough in healthcare; recall for diabetic patients is critical.

---

## 📊 Results

| Model               | Accuracy | Recall (Diabetics) | Notes                               |
| ------------------- | -------- | ------------------ | ----------------------------------- |
| Logistic Regression | \~76%    | 0.64               | Best baseline, balanced performance |
| KNN                 | \~70%    | 0.64               | Weaker, sensitive to outliers       |
| Decision Tree       | \~66%    | 0.58               | Overfitting, weaker model           |
| Random Forest       | \~75%    | 0.69               | Best recall, strong candidate       |

✅ Logistic Regression and Random Forest gave the best performance.

---

## 🚀 Future Improvements

* Apply **hyperparameter tuning** (GridSearchCV / RandomizedSearchCV).
* Handle class imbalance with **SMOTE** or `class_weight`.
* Add **ROC-AUC curves** for comparison.
* Try **Boosting algorithms** (XGBoost, LightGBM, CatBoost).
* Build a **Streamlit app** for interactive predictions.

---

## 📁 How to Run

1. Clone the repo

   ```bash
   git clone git remote add origin https://github.com/taahashahzad/Diabetes-Prediction-Project.git
   cd diabetes-prediction
   ```
2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```
3. Run Jupyter Notebook

   ```bash
   jupyter notebook
   ```

---

✨ This project helped me practice **data cleaning, EDA, scaling, model training, and evaluation**. I will revisit it in the future to improve accuracy and recall using advanced methods.

