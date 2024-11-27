import pandas as pd
import math

# This script is to add in the nearest mall/mrt to an address, as well as the distance.

def nearest_mrt(coords):
    lat1 = float(coords.split(",")[0])
    lon1 = float(coords.split(",")[1])
    min_distance = 999
    nearest = ""
    for index, i in enumerate(mrt_coords_df["coords"]):
        lat2 = float(i.split(",")[0])
        lon2 = float(i.split(",")[1])
        distance = calc_distance(lat1, lon1, lat2, lon2)
        if distance < min_distance:
            min_distance = distance
            nearest = mrt_coords_df.iloc[index]["station_name"]
    return min_distance, nearest

def nearest_mall(coords):
    lat1 = float(coords.split(",")[0])
    lon1 = float(coords.split(",")[1])
    min_distance = 999
    nearest = ""
    for index, i in enumerate(mall_coords_df["coords"]):
        lat2 = float(i.split(",")[0])
        lon2 = float(i.split(",")[1])
        distance = calc_distance(lat1, lon1, lat2, lon2)
        if distance < min_distance:
            min_distance = distance
            nearest = mall_coords_df.iloc[index]["Mall Name"]
    return min_distance, nearest


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

if __name__ == "__main__":
    main_df = pd.read_csv("Section 1 Datasets/prices_with_loc.csv")
    mrt_coords_df = pd.read_csv("Raw Datasets/mrt_lrt_data.csv")
    mrt_coords_df['coords'] = mrt_coords_df["lat"].astype(
        str) + "," + mrt_coords_df["lng"].astype(str)

    mall_coords_df=pd.read_csv("Raw Datasets/shopping_mall_coordinates.csv")
    mall_coords_df['coords'] = mall_coords_df["LATITUDE"].astype(
        str) + "," + mall_coords_df["LONGITUDE"].astype(str)
    nearest_mrt_from_coords = {}
    nearest_mall_from_coords = {}

    for i in main_df['coords'].unique():
        nearest_mrt_from_coords[i] = nearest_mrt(i)
        nearest_mall_from_coords[i] = nearest_mall(i)
    main_df["nearest_mrt"] = main_df["coords"].map(
        lambda x: nearest_mrt_from_coords[x][1])
    main_df["distance_mrt"] = main_df["coords"].map(
        lambda x: nearest_mrt_from_coords[x][0])
    main_df["nearest_mall"] = main_df["coords"].map(
        lambda x: nearest_mall_from_coords[x][1])
    main_df["distance_mall"] = main_df["coords"].map(
        lambda x: nearest_mall_from_coords[x][0])


    main_df.to_csv("Section 1 Datasets/listings with amenities.csv", index=False)
