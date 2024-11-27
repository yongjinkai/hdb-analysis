import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
df = pd.read_csv("Section 2 Datasets/A.csv")
test_df = pd.read_csv("Section 2 Datasets/example_flat_2017.csv")

X = df[['year', 'flat_type_encoded',
        'flat_age', 'town_encoded', 'floor_area_sqm', 'flat_model_encoded', 'storey_range_encoded']]
y = df['resale_price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# change X_test to this after getting key model metrics
X_test_2017 = test_df[['year', 'flat_type_encoded',
                       'flat_age', 'town_encoded', 'floor_area_sqm', 'flat_model_encoded', 'storey_range_encoded']]


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
    importance = best_model.feature_importances_
    importance_df = pd.DataFrame(importance, index=X.columns, columns=['Importance'])
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)
    
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("xgboost metrics:")
    print(f"MSE: {mse}, MAE: {mae}, R_squared: {r2}")
    # print("y_pred: ", y_pred)
    return y_pred


y_pred = xgboost()
new_df = pd.DataFrame({
    'Predicted': y_pred,
    'Actual': y_test
})

# plt.figure(figsize=(8, 6))
# sns.boxplot(data=new_df)
# plt.title('Box Plot of Predicted vs Actual Values')
# plt.xlabel('Data')
# plt.ylabel('Price')
# plt.show()
