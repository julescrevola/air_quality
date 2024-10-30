"""XGBoost regressor to predict air quality"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


# 3. Define function to train XGBoost model
def train_xgboost(X_train, y_train, preprocessor):
    # Create pipeline with preprocessor and model
    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", XGBRegressor())]
    )
    # Train model
    model.fit(X_train, y_train)
    return model


# 4. Define function to evaluate model
def evaluate_model(model, X_test, y_test):
    # Predict on test data
    y_pred = model.predict(X_test)
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse


# 5. Define function to tune hyperparameters of XGBoost model
def tune_xgboost_hyperparameters(X_train, y_train, preprocessor, param_grid=None, cv=5):
    if param_grid is None:
        # Default parameter grid if none provided
        param_grid = {
            "classifier__n_estimators": [50, 500, 1000],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
            "classifier__max_depth": [3, 5, 7],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
        }

    # Create pipeline with preprocessor and model
    model = XGBRegressor()

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        model, param_grid=param_grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Return the best model and parameters
    return grid_search.best_estimator_, grid_search.best_params_

