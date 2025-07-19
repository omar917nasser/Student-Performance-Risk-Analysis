# Student Performance Risk Analysis

A comprehensive Jupyter‑based data science project that analyzes and predicts “at‑risk” students using the Student Performance dataset. The goal is to identify students likely to underperform in the final exam and enable educational institutions to intervene early with targeted support.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Dataset](#dataset)
4. [Environment & Dependencies](#environment--dependencies)
6. [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
7. [Exploratory Data Analysis](#exploratory-data-analysis)
8. [Clustering & Risk Labeling](#clustering--risk-labeling)
10. [Classification Modeling](#classification-modeling)
11. [Model Evaluation & Comparison](#model-evaluation--comparison)
12. [Results & Visualization](#results--visualization)
13. [How to Run](#how-to-run)

---

## Project Overview

Many educational institutions struggle to identify students who are at risk of failing. By leveraging historical student performance data and modern machine learning techniques

This end‑to‑end workflow demonstrates data preprocessing, unsupervised learning, supervised classification, hyperparameter optimization, and result interpretation.

## Key Features

* **Data Cleaning Pipelines**: Automated handling of missing values, outliers, and categorical encoding
* **Clustering Analysis**: K‑Means clustering on grade patterns to derive empirical risk groups
* **Binary Risk Label**: Defined `at_risk` as true if final grade (`G3`) < 10
* **Feature Engineering**: Includes features such as grade improvements (`G2-G1`), parent involvement, and alcohol consumption
* **Multiple Classifiers**: Logistic Regression, Decision Tree, and Random Forest with rigorous `GridSearchCV`
* **Cross‑Validation Strategies**: 20‑fold to 100‑fold CV for robust performance estimates
* **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1‑score, ROC AUC, and Confusion Matrices
* **Visualization Suite**: Grade distributions, cluster plots, feature importance charts, and ROC curves in Matplotlib/Seaborn

## Dataset

* **Source**: Kaggle – Student Performance Data Set

  * [Link to Data](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set)

**Important Attributes**:

| Feature          | Type        | Description                                            |
| ---------------- | ----------- | ------------------------------------------------------ |
| `G1`, `G2`, `G3` | Numerical   | Periodic grades (0–20 scale)                           |
| `age`            | Numerical   | Student age (in years)                                 |
| `sex`            | Categorical | Student gender (`M` or `F`)                            |
| `studytime`      | Ordinal     | Weekly study time (1: <2h, 2: 2–5h, 3: 5–10h, 4: >10h) |
| `failures`       | Numerical   | Number of past class failures                          |
| `famsup`         | Categorical | Family educational support (`yes`, `no`)               |
| `absences`       | Numerical   | Number of school absences                              |
| `health`         | Ordinal     | Current health status (1: very bad – 5: very good)     |

## Environment & Dependencies

**Python Version:** 3.8+
**Notebook:** Jupyter

**Install via**:

```bash
pip install -r requirements.txt
```

**requirements.txt**:

```
pandas>=1.2
numpy>=1.20
scikit-learn>=1.0
matplotlib>=3.3
seaborn>=0.11
joblib>=1.1
```

## Data Cleaning & Preprocessing

1. **Load Data**: Read CSVs using `pandas.read_csv` with `dtype` specifications
2. **Missing Values**: No NAs in this dataset; verify with `df.isnull().sum()`
3. **Outlier Detection**: Visual inspection of `absences`, `grade` distributions; winsorize if needed
4. **Label Encoding**: Convert binary/categorical variables via `LabelEncoder` or one-hot encoding when >2 categories
5. **Normalization**: Scale continuous features (`age`, `G1`, `G2`, `absences`) using `MinMaxScaler`
6. **Target Creation**: Binary `at_risk` flag: `1` if `G3 < 10`, else `0`
7. **Train/Test Split**: 75% train / 25% test with `stratify=at_risk` and `random_state=42`

## Exploratory Data Analysis

**1. Correlation Matrix**

* Heatmap shows how grades (`G1`, `G2`, `G3`) are closely related.
* Non-academic features like `absences` show weaker links.

**2. Absences Distribution**

* Histogram highlight that most students have few absences.
* Some clear outliers may need handling.

**3. Age Distribution**

* Histogram plot shows most students are 15–17 years old.
* Distribution is narrow, so age isn't a strong differentiator.

**4. Final Grade (`G3`) Distribution**

* Box Plot to show how grades are spread.
* Helps determine pass/fail or risk thresholds.

**5. Outliers Boxplot**

* Features like `absences`, `failures`, and `studytime` show outliers.
* Useful for preprocessing decisions.

**6. Students at Risk vs. Not at Risk**

* Bar plot compares counts of students with `G3 < 10` vs. others.
* Reveals slight class imbalance.

**7. Risk Classification Plot**

* Bar plot clusters students into High, Medium, Low risk.
* Shows visual separation of risk groups.


## Clustering & Risk Labeling

* **Algorithm**: K-Means (k=3) on standardized `[G1, G2, G3]`
* **Centroid Analysis**: Sort cluster centroids by mean `G3` to map clusters → \[High, Moderate, Low] Risk

## Classification Modeling

We train and tune three algorithms:

| Model               | Hyperparameter Search                                                                      | CV Folds | Scoring    |
| ------------------- | ------------------------------------------------------------------------------------------ | -------- | ---------- |
| Logistic Regression | `penalty=[l1,l2], C=[0.01,0.1,1,10], solver=['liblinear'], class_weight=[None,'balanced']` | 20       | `f1`       |
| Decision Tree       | `max_depth=[None,5,10,15], min_samples_split=[2,5,10], criterion=['gini','entropy']`       | 100      | `accuracy` |
| Random Forest       | `n_estimators=[100,200,500], max_depth=[None,5,10], min_samples_leaf=[1,2,4]`              | 30       | `accuracy` |

Training performed with `GridSearchCV` (n\_jobs=-1) and `random_state=42` for reproducibility.

## Model Evaluation & Comparison

* **Metrics**: Accuracy, Precision, Recall, F1
* **Results Table**:

| Model               | Accuracy | Precision | Recall | F1-Score | 
| ------------------- | -------- | --------- | ------ | -------- | 
| Logistic Regression | 0.82     | 0.80      | 0.75   | 0.77     | 
| Decision Tree       | 0.78     | 0.76      | 0.70   | 0.73     | 
| Random Forest       | 0.85     | 0.83      | 0.79   | 0.81     |

*(Metrics are illustrative; see notebook for exact numbers.)*

## Results & Visualization

All figures are saved in `Visualizations/`, including:

- Correlation Matrix
- Distribution of Absences
- Distribution of Ages
- Distribution of Final Grade
- Boxplot of Outliers
- Distribution of Students at Risk vs Not at Risk
- Risk Classification
  

## How to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/student-risk-analysis.git
   cd student-risk-analysis
   ```
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. Get Dataset:
   Download Dataset: * [Link to Data](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set)

   
4. **Launch Jupyter Notebook**:

   ```bash
   jupyter notebook notebooks/Project.ipynb
   ```
