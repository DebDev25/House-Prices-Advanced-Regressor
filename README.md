# Ames Housing Price Prediction

## Overview

End-to-end house price prediction project using the Ames Housing dataset, covering structured EDA, domain-driven feature engineering, preprocessing, model training, hyperparameter optimization, and ensemble modeling.

## Dataset

The project uses the Ames Housing dataset with 80+ features describing residential properties in Ames, Iowa.

The target variable is `SalePrice`, modeled in log-transformed form using `log1p`.

## EDA

The exploratory analysis covers:

- Numeric, nominal, and ordinal feature classification
- Missing-value analysis and structural missingness
- Univariate analysis of categorical and ordinal features
- Correlation analysis and multicollinearity
- Outlier identification
- Target distribution and transformation
- Domain-driven feature analysis

## Feature Classification

Features were categorized into:

- Numeric
- Nominal categorical
- Ordinal categorical

Ordinal features were mapped according to their natural ordering for models using numerical representations.

## Missing Values

Missing values were analyzed according to their semantic meaning rather than using a single arbitrary threshold.

Structural missingness, such as the absence of a garage or pool, was treated differently from genuinely missing observations.

## Correlation and Multicollinearity

Correlation analysis was used to identify redundant and structurally related variables, including relationships such as:

- `GarageArea` and `GarageCars`
- `GrLivArea` and `TotRmsAbvGrd`
- `1stFlrSF` and `TotalBsmtSF`
- `GarageYrBlt` and `YearBuilt`

These relationships were considered during feature engineering and model selection.

## Outlier Analysis

Potentially influential observations were investigated using feature relationships and domain knowledge.

Highly unusual combinations of property size and sale price were removed where they represented clear anomalies.

## Feature Engineering

Domain-driven features were created to capture relationships between existing variables, including:

- `FootprintSize`
- `AvgRoomSize`
- `GarageAge`
- `HouseAge`
- `GarageBuiltLater`
- `AreaPerCar`

Original features were retained alongside engineered features for tree-based models.

## Target Transformation

`SalePrice` was transformed using:

```python
np.log1p(SalePrice)
```

This reduces target skewness and makes the regression problem more stable.

Predictions are converted back to the original price scale using:

```python
np.expm1(predictions)
```

## Modeling

Tree-based gradient boosting models were evaluated using cross-validation:

- XGBoost
- LightGBM
- CatBoost

XGBoost and CatBoost were selected for further hyperparameter tuning after baseline comparison.

## Hyperparameter Optimization

Optuna was used to efficiently search model hyperparameters rather than relying solely on manual or grid search.

The tuned XGBoost model achieved approximately:

**RMSE: 0.1259 on the log-transformed target**

## Ensembling

Out-of-fold predictions were used to evaluate weighted ensembles of the strongest individual models.

The final ensemble predictions are averaged in log space before being transformed back to the original price scale.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- LightGBM
- Optuna
- Matplotlib

## Project Structure

```text
Ames-Housing-Price-Prediction/
│
├── notebook.ipynb
├── README.md
└── submission.csv
```

## Evaluation

Model performance is evaluated using cross-validated RMSE on the log-transformed target.

The same cross-validation strategy is used when comparing models to maintain a fair evaluation.

## Kaggle

Final predictions are converted back to the original `SalePrice` scale and formatted as:

```text
Id,SalePrice
```

for Kaggle submission.
