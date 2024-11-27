import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

import numpy as np
df = pd.read_csv("Section 2 Datasets/A.csv")
# train data: all years excluding 2014
train_df = df[(df['year'] < 2014) | (df['year'] > 2014)].copy()
test_df_2014 = df[df['year'] == 2014].copy()  # Test data (year = 2014)

# MinMax scaling
columns_to_scale = ['town_encoded',
                    'flat_type_encoded', 'flat_age']
scaler = MinMaxScaler()
train_df[['town_scaled',
          'flat_type_scaled', 'flat_age_scaled']] = scaler.fit_transform(train_df[columns_to_scale])
test_df_2014[['town_scaled',
              'flat_type_scaled', 'flat_age_scaled']] = scaler.transform(test_df_2014[columns_to_scale])


def linear_regression():
    # Defining train and test data
    x_train = train_df[['year', 'town_scaled', 'flat_type_scaled',
                        'flat_age_scaled']]
    y_train = train_df['resale_price']
    x_test = test_df_2014[['year', 'town_scaled', 'flat_type_scaled',
                           'flat_age_scaled']]
    y_test = test_df_2014['resale_price']

    ols = LinearRegression()
    ols.fit(x_train, y_train)
    y_pred = ols.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("linear regression metrics:")
    print(f"MSE: {mse}, MAE: {mae}, R_squared: {r2} \n")
    return y_pred


def random_forest():
    X_train = train_df[['year', 'flat_type_encoded',
                        'flat_age', 'town_encoded']]
    y_train = train_df['resale_price']
    X_test = test_df_2014[[
        'year', 'flat_type_encoded', 'flat_age', 'town_encoded']]
    y_test = test_df_2014['resale_price']

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("random forest metrics:")
    print(f"MSE: {mse}, MAE: {mae}, R_squared: {r2} \n")
    return y_pred


def xgboost():
    X_train = train_df[['year', 'flat_type_encoded',
                        'flat_age', 'town_encoded']]
    y_train = train_df['resale_price']
    X_test = test_df_2014[[
        'year', 'flat_type_encoded', 'flat_age', 'town_encoded']]
    y_test = test_df_2014['resale_price']

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
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
    print("xgboost metrics:")
    print(f"MSE: {mse}, MAE: {mae}, R_squared: {r2}")

    importance = best_model.feature_importances_
    importance_df = pd.DataFrame(
        importance, index=X_test.columns, columns=['Importance'])
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)

    return y_pred


# y_pred_lg = linear_regression()
# y_pred_rf = random_forest()
y_pred_xgb = xgboost()


# For plotting predicted outcome vs actual outcome

# df['linear_reg_price'] = np.nan
# df.loc[df['year'] == 2014, 'linear_reg_price'] = y_pred_lg
# df['random_forest_price'] = np.nan
# df.loc[df['year'] == 2014, 'random_forest_price'] = y_pred_rf
# df['xgboost_price'] = np.nan
# df.loc[df['year'] == 2014, 'xgboost_price'] = y_pred_xgb
# df.to_csv('Section 2 Datasets/A.csv', index=False)
