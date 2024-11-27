import pandas as pd
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

df = pd.read_csv("Section 2 Datasets/A.csv")
df = df[df['year'] >= 2014]
df['is_yishun'] = (df['town'] == 'YISHUN').astype(int)
X = df[['year', 'flat_type_encoded',
        'flat_age', 'town_encoded', 'floor_area_sqm', 'flat_model_encoded', 'storey_range_encoded', 'is_yishun']]
y = df['resale_price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


def xgboost():

    param_grid = {

        # 'n_estimators': [100, 200],
        # 'learning_rate': [0.01, 0.05],
        # 'max_depth': [3, 5],
        # 'subsample': [0.8, 1.0]

        # Best params based on previous runs, commented out above to save time
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
    print(f"MSE: {mse}, MAE: {mae}, R_squared: {r2}")
    return y_pred


def linear_regression():
    formula = 'resale_price ~ year + flat_type_encoded + C(town) + flat_model_encoded + flat_age'
    model = smf.ols(formula=formula, data=df).fit()
    towns = df['town'].unique()
    towns.sort()
    print(towns)
    print(model.summary())


xgboost()
linear_regression()
