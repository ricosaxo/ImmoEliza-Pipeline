import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
import numpy as np
from typing import Dict, Any
import joblib 

def load_data(input_path: str, correlation_threshold: float = 0.10, file_type: str = "pickle") -> pd.DataFrame:
    """
    Load data from a file (pickle or CSV), select features based on correlation with 'Price'.

    Args:
        input_path (str): Path to the data file.
        correlation_threshold (float): Minimum correlation threshold for feature selection.
        file_type (str): Type of file to load ('pickle' or 'csv'). Default is 'pickle'.

    Returns:
        pd.DataFrame: DataFrame containing only features with correlation above the threshold with 'Price'.
    """

    # Load the data based on file type
    print(f"Loading data from: {input_path}")
    pd.set_option('display.max_columns', None)

    if file_type == "pickle":
        df = pd.read_pickle(input_path)
    elif file_type == "csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Invalid file_type specified. Use 'pickle' or 'csv'.")

    # Check for missing values
    missing_values = df.isnull().sum()
    missing_columns = missing_values[missing_values > 0]
    if not missing_columns.empty:
        print("Missing values found in the following columns:")
        print(missing_columns)
    else:
        print("No missing values found in the dataset.")

    # Exclude specific columns from processing
    columns_to_exclude = ['price_per_sqm', 'price_per_sqm_land', 'epc','Postal_code']
    df = df.drop(columns=[col for col in columns_to_exclude if col in df.columns], errors='ignore')

    # Identify non-numeric columns to be encoded later if needed
    non_numeric_columns = []
    numeric_df = pd.DataFrame()

    for col in df.columns:
        try:
            # Attempt to convert the column to a numeric type; if it fails, it is non-numeric
            numeric_df[col] = pd.to_numeric(df[col], errors='raise')
        except ValueError:
            non_numeric_columns.append(col)

    # Calculate correlations on numeric columns only
    correlation_matrix = numeric_df.corr()

    # Select features with correlation to 'Price' above the threshold
    correlated_features = correlation_matrix.index[abs(correlation_matrix['Price']) >= correlation_threshold]
    df_selected = numeric_df[correlated_features]

    print("Columns selected based on correlation threshold of", correlation_threshold, ":", list(df_selected.columns))

    # Display non-numeric columns that need encoding
    if non_numeric_columns:
        print("Non-numeric columns to encode:", non_numeric_columns)

    # Display basic information about the selected data
    print(df_selected.info())
    print(df_selected.head())
    print(df_selected.describe())

    return df_selected

def create_model_pipeline(model: Any) -> Pipeline:
    """
    Create a machine learning pipeline with standardization.
    
    Args:
        model: A scikit-learn model object
    
    Returns:
        Pipeline: Scikit-lelearn pipeline with standardization and model
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

def train_and_evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model: Any,
    model_name: str = 'Model',
    cv_splits: int = 6
) -> Dict[str, Any]:
    """
    Train and evaluate a machine learning model with cross-validation.
    
    Args:
        X: Feature matrix
        y: Target vector
        model: Scikit-learn model object
        model_name: Name of the model for logging
        cv_splits: Number of cross-validation splits
    
    Returns:
        Dict containing model and evaluation metrics
    """
    print(f"\nTraining and evaluating {model_name}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = create_model_pipeline(model)
    
    # Cross-validation
    print("Performing cross-validation...")
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_results = cross_val_score(pipeline, X_train, y_train, cv=kf)
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Generate predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'model': pipeline,
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'r2_train': r2_score(y_train, y_train_pred),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'r2_test': r2_score(y_test, y_test_pred),
        'cv_scores': cv_results
    }
    
    # Print results
    print(f"{model_name} Performance:")
    print(f"  Cross-validation scores: {cv_results}")
    print(f"  Mean CV Score: {cv_results.mean():.4f} (+/- {cv_results.std() * 2:.4f})")
    print(f"  RMSE on training data: {metrics['rmse_train']:.4f}")
    print(f"  R² Score on training data: {metrics['r2_train']:.4f}")
    print(f"  RMSE on test data: {metrics['rmse_test']:.4f}")
    print(f"  R² Score on test data: {metrics['r2_test']:.4f}\n")
    
    return metrics

def hyperparameter_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    model: Any,
    param_distributions: Dict[str, Any],
    model_name: str 
) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    
    Args:
        X: Feature matrix for training
        y: Target vector for training
        model: Scikit-learn model object
        param_distributions: Dictionary of parameter distributions for RandomizedSearchCV
        X_test: Feature matrix for testing
        y_test: Target vector for testing
        model_name: Name of the model for logging
    
    Returns:
        Dict containing best model and evaluation metrics
    """
    print(f"\nStarting hyperparameter tuning for {model_name}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = create_model_pipeline(model)
    
    rscv = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        cv=KFold(n_splits=6, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
        verbose=2
    )
    
    rscv.fit(X, y)
    
    # Evaluate best model
    y_train_pred = rscv.best_estimator_.predict(X)
    y_test_pred = rscv.best_estimator_.predict(X_test)
    
    results = {
        'best_model': rscv.best_estimator_,
        'best_params': rscv.best_params_,
        'best_score': rscv.best_score_,
        'rmse_best_train': np.sqrt(mean_squared_error(y, y_train_pred)),
        'r2_best_train': r2_score(y, y_train_pred),
        'rmse_best_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'r2_best_test': r2_score(y_test, y_test_pred)
    }
    
    print(f"{model_name} Hyperparameter Tuning Results:")
    print(f"  Best Parameters: {results['best_params']}")
    print(f"  Best Cross-Validation Score: {results['best_score']:.4f}")
    print("\nImprovement after hyperparameter tuning:")
    print(f"  RMSE with best model on training data: {results['rmse_best_train']:.4f}")
    print(f"  R² Score with best model on training data: {results['r2_best_train']:.4f}")
    print(f"  RMSE with best model on test data: {results['rmse_best_test']:.4f}")
    print(f"  R² Score with best model on test data: {results['r2_best_test']:.4f}\n")
    
    return results

def main():
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(),
        'XGBoost': xgb.XGBRegressor(eval_metric='rmse')
    }

    # Define hyperparameter distributions
    param_distributions = {
        'Linear Regression': {'model__fit_intercept': [True, False]},
        'Ridge Regression': {'model__alpha': np.linspace(0.0001,1,10), 'model__solver':['auto','svd','cholesky','lsqr','sparse_cg','sag','saga']},
        'Lasso Regression': {'model__alpha': np.logspace(-3, 3, 7),'model__fit_intercept': [True, False],'model__selection': ['random', 'cyclic']},
        'Random Forest': {'model__n_estimators': [50, 100, 200, 500],'model__max_depth': [None, 10, 20, 30, 40],'model__min_samples_split': [2, 5, 10],'model__min_samples_leaf': [1, 2, 4],'model__max_features': [None,'sqrt', 'log2'],'model__bootstrap': [True, False]},
        'XGBoost': {'model__n_estimators': [50, 100],'model__max_depth': [3, 5],'model__learning_rate': [0.1, 0.2],'model__subsample': [0.5, 0.75],'model__colsample_bytree': [0.5, 0.75]}
    }

    # Load your data
    input_path = r'output/output_preprocessing.csv'
    df_selected = load_data(input_path = input_path, file_type='csv')

    # Split data into features and target
    X = df_selected.drop(columns=['Price'])  # Assuming 'Price' is the target
    y = df_selected['Price']

    # Train and evaluate each model
    model_results = {}

    for model_name, model in models.items():
        # Train and evaluate the model
        metrics = train_and_evaluate_model(X, y, model, model_name)
        model_results[model_name] = metrics

    # Optionally, perform hyperparameter tuning for each model
    for model_name, model in models.items():
        if model_name in param_distributions:  # Ensure there are parameters defined for tuning
            results = hyperparameter_tuning(X, y, model, param_distributions[model_name], model_name=model_name)
            model_results[f"{model_name} Tuning"] = results

    # Output all model results
    for name, result in model_results.items():
        print(f"\n{name} Results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
            
    # Find the best model based on cross-validation score after tuning
    best_model_name, best_model_result = None, None
    best_cv_score = -np.inf

    # Find the best model based on tuning cross-validation scores
    for name, result in model_results.items():
        if "Tuning" in name:  # Only consider tuned models
            if result['best_score'] > best_cv_score:
                best_cv_score = result['best_score']
                best_model_name = name
                best_model_result = result

    # Display the best model found
    if best_model_result:
        print(f"\nOverall best Model: {best_model_name}")
        print("Best Parameters:", best_model_result['best_params'])
        print("Best Cross-Validation Score:", best_cv_score)

        # Make a sample prediction
        sample_index = 0  # Choose the row index you want to use for the sample
        sample_input = X.iloc[sample_index].values.reshape(1, -1)  # Reshape for single prediction

        best_model = best_model_result['best_model']
        sample_prediction = best_model.predict(sample_input)

        real_price = y.iloc[sample_index]

        print(f"\nSample Prediction with {best_model_name}:")
        print("  Input features:", X.iloc[sample_index].to_dict())
        print("  Predicted Price:", sample_prediction[0])
        print("  Actual Price:", real_price)
        print(f"  Difference (Predicted - Actual): {sample_prediction[0] - real_price:.2f}")
    else:
        print("No tuned models were found.")

        
    # Retrain the best model on the entire dataset
    if best_model_result:
        best_model = best_model_result['best_model']
        best_model.fit(X, y)  # Fit the best model on the entire dataset

        # Save the retrained model
        model_filename = 'best_trained_model.pkl'
        joblib.dump(best_model, model_filename)
        print(f"Best model retrained and saved as {model_filename}")
    else:
        print("No best model was found to retrain.")

if __name__ == "__main__":
    main()
