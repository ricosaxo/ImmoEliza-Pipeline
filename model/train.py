import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
import joblib
from typing import Dict, Any

# Define the models and hyperparameter distributions
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': xgb.XGBRegressor(eval_metric='rmse', use_label_encoder=False)
}

param_distributions = {
    'Linear Regression': {'model__fit_intercept': [True, False]},
    'Ridge Regression': {'model__alpha': np.linspace(0.0001,1,10), 'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']},
    'Lasso Regression': {'model__alpha': np.logspace(-3, 3, 7), 'model__fit_intercept': [True, False], 'model__selection': ['random', 'cyclic']},
    'Random Forest': {'model__n_estimators': [50, 100, 200, 500], 'model__max_depth': [None, 10, 20, 30, 40], 'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 4], 'model__max_features': ['auto', 'sqrt', 'log2'], 'model__bootstrap': [True, False]},
    'XGBoost': {'model__n_estimators': [50, 100], 'model__max_depth': [3, 5], 'model__learning_rate': [0.1, 0.2], 'model__subsample': [0.5, 0.75], 'model__colsample_bytree': [0.5, 0.75]}
}

def load_data(input_path: str, correlation_threshold: float = 0.2) -> pd.DataFrame:
    df = pd.read_pickle(input_path)
    df = df.select_dtypes(include=[np.number])  # Select numeric columns
    correlation_matrix = df.corr()
    correlated_features = correlation_matrix.index[abs(correlation_matrix['Price']) >= correlation_threshold]
    return df[correlated_features]

def create_model_pipeline(model: Any) -> Pipeline:
    return Pipeline([('scaler', StandardScaler()), ('model', model)])

def train_and_evaluate_model(X: pd.DataFrame, y: pd.Series, model: Any, model_name: str) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = create_model_pipeline(model)
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)
    metrics = {
        'model': pipeline,
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'r2_test': r2_score(y_test, y_test_pred),
    }
    return metrics

def hyperparameter_tuning(X: pd.DataFrame, y: pd.Series, model: Any, param_distributions: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    rscv = RandomizedSearchCV(
        create_model_pipeline(model), param_distributions=param_distributions,
        n_iter=10, cv=KFold(n_splits=6, shuffle=True, random_state=42),
        random_state=42, n_jobs=-1
    )
    rscv.fit(X, y)
    return {
        'best_model': rscv.best_estimator_,
        'best_params': rscv.best_params_,
        'best_score': rscv.best_score_,
    }

def main():
    input_path = 'data/clean/after_step_4_correlation.pkl'
    df = load_data(input_path)
    X = df.drop(columns=['Price'])
    y = df['Price']
    best_model_name, best_model, best_cv_score = None, None, -np.inf

    for model_name, model in models.items():
        tuning_results = hyperparameter_tuning(X, y, model, param_distributions[model_name], model_name=model_name)
        if tuning_results['best_score'] > best_cv_score:
            best_cv_score = tuning_results['best_score']
            best_model_name, best_model = model_name, tuning_results['best_model']
    
    # Retrain and save the best model on full dataset
    if best_model:
        best_model.fit(X, y)
        joblib.dump(best_model, 'best_trained_model.pkl')
        print(f"{best_model_name} saved as 'best_trained_model.pkl'")

if __name__ == '__main__':
    main()
