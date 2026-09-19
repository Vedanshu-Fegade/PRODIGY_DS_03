# PRODIGY_DS_03

## Prodigy InfoTech Data Science Internship – Task 03

### Task

Build a Decision Tree Classifier to predict whether a customer will subscribe to a bank term deposit using the UCI Bank Marketing dataset.

### Dataset

The project uses the **UCI Bank Marketing Dataset**.

Dataset Source:
https://archive.ics.uci.edu/dataset/222/bank+marketing

The dataset contains information about customers and marketing campaigns, with the target variable `y` indicating whether the customer subscribed to a term deposit.

### Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### Data Preprocessing

The following preprocessing steps were performed:

- The target variable `y` was converted into binary values:
  - `yes` → 1
  - `no` → 0
- The `duration` column was removed because it is only known after the campaign call and can leak target information.
- Numerical features were handled using median imputation.
- Categorical features were handled using most-frequent-value imputation.
- Categorical variables were converted using One-Hot Encoding.
- The dataset was divided into training and testing sets using an 80/20 split.
- Stratified splitting was used to preserve the target-class distribution.

### Machine Learning Model

A **Decision Tree Classifier** was used with the following configuration:

- Maximum depth: 5
- Minimum samples per leaf: 40
- Class weight: balanced
- Random state: 42

The model was implemented using a Scikit-learn Pipeline combining preprocessing and classification.

### Model Evaluation

The model was evaluated using:

- Confusion Matrix
- Classification Report
- F1 Score

### Results

The model was evaluated on the test dataset.

The generated confusion matrix contains:

| Actual / Predicted | No | Yes |
|---|---:|---:|
| No | 6596 | 1389 |
| Yes | 464 | 594 |

The model achieved an F1 score of approximately **0.391** for the positive subscription class.

### Visualizations

The project generates the following visualizations:

#### 1. Confusion Matrix

`outputs/confusion_matrix.png`

The confusion matrix shows the number of correct and incorrect predictions for customers who did and did not subscribe to the term deposit.

#### 2. Feature Importance

`outputs/feature_importance.png`

This visualization shows the most important features used by the Decision Tree Classifier based on impurity-based feature importance.

### Project Structure

```text
PRODIGY_DS_03/
│
├── data/
│   └── bank-full.csv
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── metrics.txt
│
├── task03.py
├── README.md
└── requirements.txt
