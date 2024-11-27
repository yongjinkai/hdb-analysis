import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import shap
df = pd.read_csv("Section 2 Datasets/A.csv")
df = df[df['lease_commence_date'] >= 2000]

X = df[['lease_commence_date', 'year', 'flat_type_encoded',
        'flat_age', 'town_encoded',  'flat_model_encoded', 'storey_range_encoded', 'resale_price']]
y = df['floor_area_sqm']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


def xgboost():

    param_grid = {
        # Using best parameters based on previous runs, commented out test params to save time
        # 'n_estimators': [100, 200],
        # 'learning_rate': [0.01, 0.05],
        # 'max_depth': [3, 5],
        # 'subsample': [0.8, 1.0],
        'n_estimators': [200],
        'learning_rate': [0.05],
        'max_depth': [5],
        'subsample': [0.8]
    }
    grid_search = GridSearchCV(estimator=xgb.XGBRegressor(
        objective='reg:squarederror'), param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    importance = best_model.feature_importances_
    importance_df = pd.DataFrame(
        importance, index=X.columns, columns=['Importance'])
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)
    print("xgboost metrics:")
    print(f"R_squared: {r2}")

    explainer = shap.TreeExplainer(best_model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_train)

    # Visualize SHAP values for `lease_commence_date` (dependence plot)
    shap.dependence_plot('lease_commence_date', shap_values, X_train,)
    return y_pred


y_pred = xgboost()
