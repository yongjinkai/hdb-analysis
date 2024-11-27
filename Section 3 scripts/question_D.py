import pandas as pd
from sklearn.linear_model import LinearRegression
import math
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import statsmodels.formula.api as smf


# Data preparation
raw_df = pd.read_csv("Section 1 Datasets/prices_with_loc.csv")
mrt_lrt_df = pd.read_csv(
    "Raw Datasets/mrt_lrt_data.csv", index_col='station_name')
CITY_CENTER = [mrt_lrt_df.loc['Raffles Place']
               ['lat'], mrt_lrt_df.loc['Raffles Place']['lng']]


def calc_distance(lat1, lon1, lat2, lon2):

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Approximating distance
    x = dlon * math.cos((lat1 + lat2) / 2)
    y = dlat
    R = 6371  # Earth's radius in kilometers

    # Distance in kilometers
    distance = R * math.sqrt(x**2 + y**2)
    return distance


def distance_to_center():
    coords_to_dist = {}
    lat2 = CITY_CENTER[0]
    lon2 = CITY_CENTER[1]
    for i in raw_df['coords'].unique():
        lat1 = float(i.split(',')[0])
        lon1 = float(i.split(',')[1])
        distance = calc_distance(lat1, lon1, lat2, lon2)
        coords_to_dist[i] = distance
    raw_df['distance_to_center'] = raw_df['coords'].map(coords_to_dist)
    raw_df.to_csv("Section 3 Datasets/D.csv")


# distance_to_center()
# merging original dataframe with COE prices
df = pd.read_csv("Section 3 Datasets/D.csv")
df = df[['month', 'town', 'flat_type', 'floor_area_sqm',
         'flat_model', 'resale_price', 'distance_to_center']]
df['month'] = pd.to_datetime(df['month'])
coe_df = pd.read_csv("Raw Datasets/coe_results.csv")
coe_df.columns = coe_df.columns = ["Date", "Cat A", "Cat B"]
coe_df['coe_price'] = (coe_df['Cat A'] + coe_df['Cat B'])/2
coe_df['Date'] = pd.to_datetime(
    coe_df['Date'], dayfirst=True).dt.to_period('M')

coe_df = coe_df.groupby('Date').agg('first')
coe_df.index = coe_df.index.to_timestamp()
merged = pd.merge(left=df, right=coe_df,  how='inner',
                  right_index=True, left_on='month')
merged = merged.sort_values(by='month')

# Additional dataset creations for testing purposes
analysis_towns = ['CENTRAL AREA', 'JURONG WEST',
                  'PASIR RIS', 'WOODLANDS', 'PUNGGOL']
further_towns = ['JURONG WEST',
                 'PASIR RIS', 'WOODLANDS', 'PUNGGOL']
analysis_towns_df = merged[merged['town'].isin(analysis_towns)].copy()
analysis_towns_df['further'] = analysis_towns_df['town'].apply(
    lambda x: 1 if x in further_towns else 0)
analysis_towns_df['coe_price_shifted'] = analysis_towns_df['coe_price'].shift(
    1)
analysis_towns_df_shifted = analysis_towns_df.dropna().copy()


# encoding
encoder = TargetEncoder(cols=['town', "flat_type", "flat_model"])
encoded = encoder.fit_transform(
    merged[["town", "flat_type", 'flat_model']], merged['resale_price'])
merged[['town_encoded', 'flat_type_encoded', 'flat_model_encoded']] = encoded

le = LabelEncoder()
merged['month_encoded'] = le.fit_transform(merged['month'])


# Standard Scaling df for testing
std_scaled = StandardScaler().fit_transform(
    merged[['floor_area_sqm', 'town_encoded', 'flat_type_encoded', 'flat_model_encoded', 'resale_price', 'distance_to_center', 'coe_price']])
std_scaled_df = merged.copy()
std_scaled_df[['floor_area_sqm', 'town_encoded', 'flat_type_encoded',
               'flat_model_encoded', 'resale_price', 'distance_to_center', 'coe_price']] = std_scaled

# MinMax Scaling df for testing
mmscaler = MinMaxScaler()
mm_scaled = mmscaler.fit_transform(merged[['floor_area_sqm', 'town_encoded', 'flat_type_encoded',
                                   'flat_model_encoded', 'resale_price', 'distance_to_center', 'coe_price']])
mm_scaled_df = merged.copy()
mm_scaled_df[['floor_area_sqm', 'town_encoded', 'flat_type_encoded',
              'flat_model_encoded', 'resale_price', 'distance_to_center', 'coe_price']] = mm_scaled

# 2 formulas, 1 with distance as binary(near/far), the other one as continuous
formula1 = 'resale_price ~  month_encoded + floor_area_sqm + flat_model_encoded + distance_to_center + coe_price + distance_to_center:coe_price'
formula2 = 'resale_price ~  month_encoded + floor_area_sqm + flat_model_encoded + further + coe_price + further:coe_price'


model = smf.ols(formula=formula2, data=analysis_towns_df).fit()
print(model.summary())
