# Student Performance Risk Analysis

A Jupyter‑based data science project that analyzes and predicts “at‑risk” students using the UCI Student Performance dataset. We clean and preprocess the data, perform clustering to label risk levels, then train and tune several classification models to identify students likely to underperform.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Environment & Dependencies](#environment--dependencies)  
4. [Project Structure](#project-structure)  
5. [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)  
6. [Exploratory Analysis & Clustering](#exploratory-analysis--clustering)  
7. [Classification Modeling](#classification-modeling)  
8. [Results & Evaluation](#results--evaluation)  
9. [How to Run](#how-to-run)  
10. [Contributing](#contributing)  
11. [License](#license)  

---

## Project Overview

Many educational institutions want to identify students at risk of failing so that they can intervene early. In this project, we:

- Load the UCI secondary‑school student performance dataset (monthly grades, demographic, social and school data)  
- Clean, encode, and normalize features  
- Use **K‑Means** clustering to define risk groups (“High Risk”, “Moderate Risk”, “Low Risk”) based on grade patterns  
- Train and tune three classifiers—**Logistic Regression**, **Decision Tree**, and **Random Forest**—to predict whether a student is “at risk” of scoring below 10 in the final exam  
- Evaluate model performance using cross‑validation, accuracy, F1, confusion matrices, and classification reports  

---

## Dataset

- **Source:** UCI Machine Learning Repository – Student Performance Data Set  
- **Files used:**  
  - `student-por.csv` (Portuguese language course data)  
  - *Note:* place your CSV in `data/` or adjust the path accordingly in the notebook.

**Key variables**:

- `G1`, `G2`, `G3`: grades at the first, second, and final period (0–20 scale)  
- Demographics: `age`, `sex`, `address`, `famsize`, `Pstatus`, …  
- Family and school-related features: parental education (`Medu`, `Fedu`), study time, failures, extracurriculars, alcohol use, etc.

---

## Environment & Dependencies

This project was developed and tested with:

- Python 3.8+  
- Jupyter Notebook  
- pandas  
- numpy  
- seaborn  
- matplotlib  
- scikit‑learn  

You can install all dependencies via:

```bash
pip install -r requirements.txt
****
