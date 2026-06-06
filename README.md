# Credit Scoring Model Using Machine Learning

# Author
Shaik Nazma

# Internship
CodeAlpha Machine Learning Internship

## Project Overview

This project predicts an individual's creditworthiness using historical financial information and machine learning techniques. The model analyzes factors such as income, loan amount, credit history, education, employment status, and property area to determine whether a loan application is likely to be approved.

## Objective

The objective of this project is to build and evaluate machine learning models that can accurately predict loan approval status based on applicant financial and demographic information.

## Dataset

The project uses the Loan Prediction Dataset containing applicant financial details and loan approval information.

### Dataset Features

* Gender
* Married Status
* Dependents
* Education
* Self Employed
* Applicant Income
* Coapplicant Income
* Loan Amount
* Loan Amount Term
* Credit History
* Property Area

### Target Variable

* Loan Status (Approved / Not Approved)

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Machine Learning Models

### Logistic Regression

Used as a baseline classification model to predict loan approval status.

### Random Forest Classifier

Used to improve prediction performance through ensemble learning.

## Data Preprocessing

* Missing value handling
* Categorical data encoding using Label Encoder
* Feature scaling using StandardScaler
* Train-Test Split (80:20)

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

## Results

### Logistic Regression

* Accuracy: 78.86%

### Random Forest

* Accuracy: 74.80%

### ROC-AUC Score

* 0.672

## Visualizations

* Confusion Matrix Heatmap
* Accuracy Comparison Graph

## Project Structure

Credit_Scoring_Model

├── main.py

├── train_u6lujuX_CVtuZ9i.csv

├── README.md

├── requirements.txt

├── screenshots

│   ├── terminal_output.png

│   ├── confusion_matrix.png

│   └── accuracy_comparison.png

└── report

    └── Credit_Scoring_Report.docx

## Conclusion

The project successfully predicts loan approval status using machine learning algorithms. Logistic Regression achieved the best performance with an accuracy of 78.86%, while Random Forest achieved 74.80%. The results demonstrate how machine learning can support credit risk assessment and loan approval decisions.
