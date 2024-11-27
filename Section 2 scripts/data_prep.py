import pandas as pd
from category_encoders import TargetEncoder

# Mock dataframe for Question B
flat_data = {
    'flat_type': ['4 ROOM'],
    'town': ['Yishun'],
    'flat_model': ['New Generation'],
    'storey_range': ['10 TO 12'],
    'floor_area_sqm': [91],
    'lease_commence_date': [1984],
    'year': [2017],
    'resale_price': [550800]
}
flat_data_df = pd.DataFrame(flat_data)

# Data columns preparation
df = pd.read_csv("Section 2 Datasets/merged_prices_updated.csv")
df['year'] = df['month'].apply(lambda x: int(x[:4]))
df['flat_age'] = df['year'] - df['lease_commence_date']
flat_data_df['flat_age'] = flat_data_df['year'] - flat_data_df['lease_commence_date']

# target encoding for original and mock df
encoder = TargetEncoder(cols=['town', "flat_type", "flat_model"])
encoded = encoder.fit_transform(
    df[["town", "flat_type", 'flat_model']], df['resale_price'])
df[['town_encoded', 'flat_type_encoded', 'flat_model_encoded']] = encoded
df = df.sort_values(by='year')
encoded_flat_data = encoder.transform(
    flat_data_df[["town", "flat_type", 'flat_model']], flat_data_df['resale_price'])
flat_data_df[['town_encoded', 'flat_type_encoded',
              'flat_model_encoded']] = encoded_flat_data

# Label encoding for original and mock df
storey_order = df['storey_range'].sort_values().unique()
storey_mapping = {x: label for label, x in enumerate(storey_order)}
df['storey_range_encoded'] = df['storey_range'].map(
    storey_mapping)

flat_data_df['storey_range_encoded'] = flat_data_df['storey_range'].map(
    storey_mapping)


df.to_csv("Section 2 Datasets/A.csv", index=False)
flat_data_df.to_csv("Section 2 Datasets/example_flat_2017.csv", index=False)
