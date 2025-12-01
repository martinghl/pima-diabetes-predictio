## Step‑by‑Step Analysis of the Pima Indians Diabetes Dataset

The following sections document the complete workflow used to explore and model the Pima Indians Diabetes dataset.  Each step includes the Python code that was executed and the resulting outputs.  You can reproduce these results by running the code in a Python environment with the `pandas`, `numpy`, and `scikit‑learn` libraries installed.

### 1. Loading the dataset

```python
import pandas as pd

# Column names based on the data description
column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# Load the CSV file from the local path
df = pd.read_csv('pima-indians-diabetes.csv', header=None, names=column_names)

print('Dataset shape:', df.shape)
print(df.head())
```

**Output**

```
Dataset shape: (768, 9)
   Pregnancies  Glucose  BloodPressure  ...  DiabetesPedigreeFunction  Age  Outcome
0            6      148             72  ...                     0.627   50        1
1            1       85             66  ...                     0.351   31        0
2            8      183             64  ...                     0.672   32        1
3            1       89             66  ...                     0.167   21        0
4            0      137             40  ...                     2.288   33        1
```  

The dataset contains 768 records (rows) and 9 columns.  Each row corresponds to one patient.  The first eight columns are clinical measurements and the final column (`Outcome`) indicates whether the patient was diagnosed with diabetes (`1`) or not (`0`).

### 2. Outcome distribution

```python
print(df['Outcome'].value_counts())
```

**Output**

```
Outcome
0    500
1    268
Name: count, dtype: int64
```

There are 500 non‑diabetic cases and 268 diabetic cases, showing a class imbalance (approximately 65 % vs 35 %).

### 3. Descriptive statistics

```python
print(df.describe())
```

**Output (summary)**

The table below shows the mean, standard deviation and quartiles of each feature before any cleaning.  Zero values are treated as valid numbers at this stage, even though they may represent missing measurements.

| Feature                   | Mean  | Std Dev | Min | 25%  | 50%  | 75%  | Max |
|--------------------------|------:|-------:|----:|-----:|-----:|-----:|----:|
| Pregnancies              | 3.85  | 3.37    | 0   | 1.0  | 3.0  | 6.0  | 17  |
| Glucose                  | 120.89| 31.97   | 0   | 99   | 117  | 140.25| 199 |
| BloodPressure            | 69.11 | 19.36   | 0   | 62   | 72   | 80   | 122 |
| SkinThickness            | 20.54 | 15.95   | 0   | 0    | 23   | 32   | 99  |
| Insulin                  | 79.80 | 115.24  | 0   | 0    | 30.5 | 127   | 846 |
| BMI                      | 31.99 | 7.88    | 0   | 27.3 | 32.0 | 36.6 | 67.1|
| DiabetesPedigreeFunction | 0.47  | 0.33    | 0.08| 0.24 | 0.37 | 0.63 | 2.42|
| Age                      | 33.24 | 11.76   | 21  | 24   | 29   | 41   | 81  |
| Outcome                  | 0.35  | 0.48    | 0   | 0    | 0    | 1    | 1   |

### 4. Handling missing values

Domain knowledge indicates that some variables cannot legitimately be zero; zeros therefore represent missing measurements.  We replace zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` with `NaN` and then impute missing values using the median of each column.

```python
import numpy as np

df_clean = df.copy()
missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in missing_cols:
    df_clean.loc[df_clean[col] == 0, col] = np.nan
    median_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_val)

print('Missing counts after replacement:')
print(df_clean.isnull().sum())
```

**Output**

```
Missing counts after replacement:
Pregnancies                   0
Glucose                       0
BloodPressure                 0
SkinThickness                 0
Insulin                       0
BMI                           0
DiabetesPedigreeFunction      0
Age                           0
Outcome                       0
dtype: int64
```

After imputation, no missing values remain in the cleaned dataset.  Descriptive statistics for the cleaned data are similar to the original ones, except that the minimum values for the imputed columns are now realistic (e.g., minimum `Glucose` value is 44 instead of 0).

### 5. Correlation analysis

We compute the Pearson correlation matrix to understand linear relationships between features and the outcome.

```python
correlation_matrix = df_clean.corr()
print(correlation_matrix)
```

**Output (excerpt)**

|                      | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI  | DiabetesPedigreeFunction | Age   | Outcome |
|----------------------|------------:|--------:|--------------:|--------------:|-------:|-----:|--------------------------:|------:|--------:|
| **Pregnancies**      | 1.000       | 0.128   | 0.209         | 0.082         | 0.025  | 0.022| –0.034                   | 0.544 | 0.222   |
| **Glucose**          | 0.128       | 1.000   | 0.219         | 0.193         | 0.419  | 0.231|  0.137                   | 0.267 | 0.493   |
| **BloodPressure**    | 0.209       | 0.219   | 1.000         | 0.196         | 0.082  | 0.283|  0.042                   | 0.325 | 0.166   |
| **BMI**              | 0.022       | 0.231   | 0.283         | 0.306         | 0.479  | 1.000|  0.173                   | 0.026 | 0.312   |
| **Outcome**          | 0.222       | 0.493   | 0.166         | 0.215         | 0.204  | 0.312|  0.174                   | 0.238 | 1.000   |

Positive correlations (e.g., between **Glucose** and **Outcome**) suggest that higher glucose levels are associated with a higher probability of diabetes.  The correlation between **Age** and **Pregnancies** is also notable (`0.544`), reflecting that older participants tend to have had more pregnancies.

### 6. Splitting and scaling the data

We split the cleaned data into a training set (80 %) and a test set (20 %).  Features are standardised to zero mean and unit variance, which is important for algorithms like logistic regression.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 7. Training the models

We train two models: a logistic regression (a linear classifier that estimates log‑odds of the outcome) and a random forest (an ensemble of decision trees that captures non‑linear relationships).

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Instantiate models
log_reg = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)

# Fit models to the scaled training data
log_reg.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)
```

### 8. Evaluating model performance

We evaluate each model on the test set using accuracy, precision, recall, F1 score and the confusion matrix.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Logistic regression predictions and metrics
y_pred_log = log_reg.predict(X_test_scaled)
metrics_log = {
    'accuracy': accuracy_score(y_test, y_pred_log),
    'precision': precision_score(y_test, y_pred_log),
    'recall': recall_score(y_test, y_pred_log),
    'f1': f1_score(y_test, y_pred_log),
    'confusion_matrix': confusion_matrix(y_test, y_pred_log)
}

# Random forest predictions and metrics
y_pred_rf = rf.predict(X_test_scaled)
metrics_rf = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf),
    'recall': recall_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'confusion_matrix': confusion_matrix(y_test, y_pred_rf)
}

print('Logistic Regression metrics:', metrics_log)
print('Random Forest metrics:', metrics_rf)
```

**Output**

```
Logistic Regression metrics: {
  'accuracy': 0.708,
  'precision': 0.600,
  'recall': 0.500,
  'f1': 0.545,
  'confusion_matrix': array([[82, 18],
                             [27, 27]])
}

Random Forest metrics: {
  'accuracy': 0.740,
  'precision': 0.652,
  'recall': 0.556,
  'f1': 0.600,
  'confusion_matrix': array([[84, 16],
                             [24, 30]])
}
```

The random forest model slightly outperforms logistic regression on all metrics.  The confusion matrices show that both models struggle to correctly classify some diabetic cases (bottom row), reflecting the class imbalance.

### 9. Additional models: XGBoost and a neural network

To broaden the comparison beyond linear and tree‑based models, we trained two more algorithms on the same preprocessed data:

* **XGBoost** – an efficient gradient‑boosting method that builds an ensemble of weak learners (decision trees) and optimises them sequentially.  It is known for strong performance on tabular data.
* **Neural network** – a simple multi‑layer perceptron (MLP) implemented via scikit‑learn’s `MLPClassifier`, comprising two hidden layers with 50 neurons each.  This provides a basic feed‑forward neural network without resorting to external deep‑learning libraries.

We used the same training/test split and scaled features as before.  The following code shows how each model was trained and evaluated:

```python
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# XGBoost classifier configuration
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Neural network (MLP with two hidden layers)
mlp_model = MLPClassifier(
    hidden_layer_sizes=(50, 50),
    activation='relu',
    max_iter=1000,
    random_state=42
)
mlp_model.fit(X_train_scaled, y_train)
y_pred_mlp = mlp_model.predict(X_test_scaled)

# Compute evaluation metrics for both models
metrics_xgb = {
    'accuracy': accuracy_score(y_test, y_pred_xgb),
    'precision': precision_score(y_test, y_pred_xgb),
    'recall': recall_score(y_test, y_pred_xgb),
    'f1': f1_score(y_test, y_pred_xgb),
    'confusion_matrix': confusion_matrix(y_test, y_pred_xgb)
}
metrics_mlp = {
    'accuracy': accuracy_score(y_test, y_pred_mlp),
    'precision': precision_score(y_test, y_pred_mlp),
    'recall': recall_score(y_test, y_pred_mlp),
    'f1': f1_score(y_test, y_pred_mlp),
    'confusion_matrix': confusion_matrix(y_test, y_pred_mlp)
}

print('XGBoost metrics:', metrics_xgb)
print('Neural network metrics:', metrics_mlp)

## Cross‑validation for the additional models
X_scaled_full = scaler.fit_transform(X)
cv_xgb = cross_val_score(
    XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42
    ),
    X_scaled_full, y,
    cv=10, scoring='accuracy'
)
cv_mlp = cross_val_score(
    MLPClassifier(hidden_layer_sizes=(50, 50), activation='relu', max_iter=1000, random_state=42),
    X_scaled_full, y,
    cv=10, scoring='accuracy'
)
print('XGBoost cross‑val mean accuracy:', cv_xgb.mean())
print('Neural network cross‑val mean accuracy:', cv_mlp.mean())
```

**Output**

```
XGBoost metrics: {
  'accuracy': 0.760,
  'precision': 0.667,
  'recall': 0.630,
  'f1': 0.648,
  'confusion_matrix': array([[83, 17],
                             [20, 34]])
}

Neural network metrics: {
  'accuracy': 0.753,
  'precision': 0.648,
  'recall': 0.648,
  'f1': 0.648,
  'confusion_matrix': array([[81, 19],
                             [19, 35]])
}

XGBoost cross‑val mean accuracy: 0.762
Neural network cross‑val mean accuracy: 0.702
```

Among all four models, XGBoost achieved the highest test accuracy and F1 score and maintained strong performance in cross‑validation.  The neural network performed comparably on the test set but showed a lower cross‑validated accuracy (~70 %), suggesting that it may over‑fit this small dataset without further tuning.

### 10. Cross‑validation

To assess how well the models generalise, we perform 10‑fold cross‑validation on the entire cleaned dataset (after scaling).  Cross‑validation splits the data into 10 subsets, trains the model on 9 subsets and tests on the remaining one, repeating this process so each subset is used for validation once.

```python
from sklearn.model_selection import cross_val_score

X_scaled_full = scaler.fit_transform(X)

cv_log = cross_val_score(
    LogisticRegression(max_iter=1000, random_state=42),
    X_scaled_full, y,
    cv=10, scoring='accuracy'
)

cv_rf = cross_val_score(
    RandomForestClassifier(n_estimators=200, random_state=42),
    X_scaled_full, y,
    cv=10, scoring='accuracy'
)

print('Logistic Regression cross‑val mean accuracy:', cv_log.mean())
print('Random Forest cross‑val mean accuracy:', cv_rf.mean())
```

**Output**

```
Logistic Regression cross‑val mean accuracy: 0.767
Random Forest cross‑val mean accuracy: 0.758
```

The cross‑validation results confirm that both models achieve moderate accuracy (around **76 %**) across different data splits.  The logistic regression slightly edges the random forest in cross‑validated accuracy, although the differences are small.

### 11. Summary of findings

1. **Exploratory analysis** showed that the dataset contains 768 entries with a notable imbalance between non‑diabetic and diabetic cases (65 % vs 35 %).  Several features, especially **Glucose**, **BMI**, **Age** and **Pregnancies**, exhibit moderate correlations with diabetes outcomes.
2. **Data cleaning** addressed missing values represented by zero entries in physiological measurements.  Imputation with column medians ensured no missing values remained.
3. **Model training and evaluation** compared a linear model (logistic regression) with a non‑linear ensemble (random forest).  The random forest achieved slightly higher test performance, but cross‑validation suggested similar overall accuracy for both models (~0.76).
4. **Additional models** extended the comparison with XGBoost and a multi‑layer perceptron.  XGBoost achieved the highest test accuracy and balanced precision/recall, while the neural network performed comparably on the test set but showed lower cross‑validated accuracy (~0.70), indicating potential over‑fitting.
5. **Benchmarking** against published studies shows that advanced methods (deep learning architectures and oversampling techniques) can achieve significantly higher accuracy (up to ~98 %) on this dataset【401310684704770†L122-L156】.  Our results serve as a transparent baseline and highlight the importance of data preprocessing, model selection and class‑imbalance handling in medical prediction problems.
