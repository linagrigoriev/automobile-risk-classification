# Automobile Risk Classification - Data Science Final Project

This project involves building a classification model to predict automobile risk levels based on the UCI "Automobile" dataset. The main objective is to predict the `Symboling` value, which reflects the riskiness of insuring a particular car.

---

## Dataset

- **Name**: Automobile Risk Evaluation
- **Source**: [KEEL Dataset Repository](https://sci2s.ugr.es/keel/dataset_smja.php?cod=1459)
- **Records**: 205
- **Features**: 26 (mixed categorical and numerical)
- **Target**: `Symboling` (insurance risk rating)

---

## Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- Visualized missing values using heatmaps.
- Plotted histograms for numerical features and bar plots for categorical ones.
- Detected missing data in both numerical and categorical columns.

### 2. Data Cleaning & Imputation
- Used `RandomForestRegressor` to predict missing values in numerical columns.
- Applied mode imputation for missing categorical fields.

### 3. Feature Engineering
- Scaled numeric features using `StandardScaler`.
- Applied **One-Hot Encoding** to categorical variables.
- Plotted correlation matrix for feature insights.

### 4. Data Splitting & Balancing
- Split the dataset into training and test sets (80/20).
- Used **SMOTE** (Synthetic Minority Over-sampling Technique) to balance classes.

### 5. Model Training
Trained and compared three models:
- `RandomForestClassifier`
- `KNeighborsClassifier`
- `LogisticRegression`

Evaluated all models using:
- Accuracy
- Precision
- Recall
- F1-score

### 6. Hyperparameter Tuning
- Optimized `KNeighborsClassifier` using `RandomizedSearchCV`.
- Achieved perfect classification accuracy post-tuning.

---

## Results

| Model                | Accuracy |
|---------------------|----------|
| Random Forest        | 1.00     |
| K-Nearest Neighbors  | 0.83 (before) → 1.00 (after tuning) |
| Logistic Regression  | 0.88     |

> Note: Due to the small dataset size and class imbalance, perfect accuracy may not reflect true generalization performance.

---

## Discussion

- **Random Forest** performed best, likely due to its robustness to feature types and overfitting risk.
- **KNN** required tuning but reached excellent performance.
- **Logistic Regression** performed reasonably well but was slightly less accurate on minority classes.
- More balanced and larger datasets would improve the reliability of the models' performance evaluations.
