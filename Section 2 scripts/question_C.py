import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Section 2 Datasets/A.csv')
data = data[data['year'] >= 2014]
X = data[['year', 'town_encoded', 'storey_range_encoded',
          'floor_area_sqm', 'resale_price', 'flat_age', 'flat_model_encoded']]
y = data['flat_type']
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
feature_importances = pd.DataFrame(rf_classifier.feature_importances_,
                                   index=X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
feature_importances['importance'] = (feature_importances['importance'])
print(feature_importances)

new_df = pd.DataFrame({
    'Predicted': y_pred,
    'Actual': y_test
})
# new_df.to_csv("Section 2 Datasets/accuracy_C.csv")
