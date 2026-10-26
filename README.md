# Ames Housing Price Prediction
 
## Overview
 
End-to-end house price prediction project using the Ames Housing dataset, covering structured EDA, domain-driven feature engineering, leak-free preprocessing, model training, hyperparameter optimization, and ensemble modeling.
 
## Dataset
 
The project uses the Kaggle *House Prices: Advanced Regression Techniques* data: 1,460 training homes and 1,459 test homes in Ames, Iowa, each described by 79 features.
 
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
 
- Numeric (33)
- Nominal categorical (28), including `MSSubClass`, which is stored as a number but is a dwelling-type code
- Ordinal categorical (18), such as quality ratings `Po < Fa < TA < Gd < Ex`

Ordinal features are mapped to integers according to their natural ordering, so tree models can split on "good or better" in a single cut.
 
## Missing Values
 
Missing values were analyzed according to their semantic meaning rather than using a single arbitrary threshold.
 
Structural missingness, such as the absence of a garage, alley or pool, is treated as information rather than a gap:
 
- Nominal columns where a missing value means "none" (`Alley`, `Fence`, `MiscFeature`, `GarageType`, `GarageFinish`, `MasVnrType`) are filled with the category `"None"`.
- Areas and counts that are missing because the feature is absent (veneer and basement areas, basement bathrooms) are filled with `0`.
- Ordinal columns map a missing value to `"None"` → `0` on their scale.

Genuinely missing values (`LotFrontage`, `Electrical`) are imputed inside the model pipeline. The same fills are applied to the test set.
 
## Correlation and Multicollinearity
 
Correlation analysis was used to identify redundant and structurally related variables (|r| > 0.7), including:
 
- `GarageArea` and `GarageCars` (0.88)
- `GarageYrBlt` and `YearBuilt` (0.83)
- `GrLivArea` and `TotRmsAbvGrd` (0.83)
- `1stFlrSF` and `TotalBsmtSF` (0.82)

These pairs were used to engineer combined features rather than simply dropping one of each pair.
 
## Outlier Analysis
 
Potentially influential observations were investigated using feature relationships and domain knowledge.
 
Two houses with more than 4,000 sq ft of living area that sold for under $200,000 were removed from the training data. These are the partial sales that the dataset's author recommends removing. Additional filters on garage and basement size were evaluated but not applied: they mostly caught the same house plus two others with no documented anomaly, and they slightly worsened cross-validated error.
 
## Feature Engineering
 
Domain-driven features were created from the correlated pairs:
 
- `FootprintSize` = max(`1stFlrSF`, `TotalBsmtSF`)
- `AvgRoomSize` = `GrLivArea` / `TotRmsAbvGrd`
- `HouseAge` = `YrSold` − `YearBuilt`
- `GarageAge` = `YrSold` − `GarageYrBlt`
- `GarageBuiltLater` = `GarageYrBlt` > `YearBuilt`
- `AreaPerCar` = `GarageArea` / (`GarageCars` + 1)

Original features are retained alongside the engineered ones, and the same features are computed for the test set.
 
## Target Transformation
 
`SalePrice` is right-skewed (skew ≈ 1.88), so it is transformed using:
 
```python
np.log1p(SalePrice)
```
 
This makes errors relative rather than dominated by expensive houses, and RMSE on the log target equals the competition metric (RMSLE).
 
Predictions are converted back to the original price scale using:
 
```python
np.expm1(predictions)
```
 
## Preprocessing
 
For XGBoost and LightGBM, preprocessing is a scikit-learn `ColumnTransformer` inside a `Pipeline`, so every imputer and encoder is refit on each training fold and no information leaks from the validation fold:
 
- Numeric: median imputation
- Nominal: most-frequent imputation, then one-hot encoding (unseen categories ignored)
- Ordinal: a custom `OrdinalMapper` that maps each category to its rank

CatBoost receives the raw columns instead: nominal features are passed as native categorical features (encoded with ordered target statistics), and ordinal features as integers so their order is kept.
 
## Modeling
 
Three gradient boosting models were compared on the same shuffled 5-fold split (seed 42):
 
- XGBoost
- LightGBM
- CatBoost

XGBoost and CatBoost were selected for hyperparameter tuning after the baseline comparison.
 
## Hyperparameter Optimization
 
Optuna's TPE sampler (seeded for reproducibility) was used to tune each model for 30 trials on the same 5 folds.
 
The best single model is the tuned **CatBoost**:
 
| Parameter | Value |
|---|---|
| `iterations` | 1028 |
| `learning_rate` | 0.058 |
| `depth` | 6 |
| `l2_leaf_reg` | 2.30 |
 
**Cross-validated RMSE: 0.1144 on the log-transformed target**, meaning predictions are typically within about 12% of the actual sale price.
 
## Ensembling
 
Out-of-fold (OOF) predictions are generated for the tuned XGBoost and CatBoost models on the shared folds, so every training house is predicted by models that never saw it. The blend weight is chosen by minimizing OOF RMSE, the predictions are averaged in log space, and the result is transformed back to the price scale.
 
The notebook reports single-model and ensemble OOF RMSE side by side, along with the correlation of the two models' errors.
 
## Evaluation
 
Model performance is evaluated using cross-validated RMSE on the log-transformed target. The same folds are used for baselines, tuning and the ensemble so that every comparison is fair.
 
Note: removing outliers also removes them from the validation folds, so part of the improvement over earlier runs reflects easier folds rather than a better model.
 
## Kaggle Submission
 
Final models are retrained on all training data and predictions are converted back to the original `SalePrice` scale in the format:
 
```text
Id,SalePrice
```
 
The notebook writes:
- `submission_xgb.csv`: tuned XGBoost alone
- `submission_ensemble.csv`: XGBoost + CatBoost blend
- `submission.csv`: whichever of the two has the lower OOF RMSE

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
House-Prices-Advanced-Regressor/
│
├── House_Price_Prediction_updated.ipynb
├── data_description.txt
├── categorical_univariate_analysis.txt
├── ordinal_univariate_analysis.txt
└── README.md
```
 
## Running
 
Run the notebook on Kaggle with the competition data attached, or place `train.csv` and `test.csv` next to the notebook. Submission files are written to `/kaggle/working/` on Kaggle, or to the notebook's folder locally.
 

Claude finished the response











