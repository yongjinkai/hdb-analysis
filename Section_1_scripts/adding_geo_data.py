import requests
import pandas as pd
from tqdm import tqdm

# This script is to add in latitude/longitude info to our dataset

df = pd.read_csv("Section 1 Datasets/merged_prices.csv")

def get_geom(address):
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={address}&returnGeom=Y&getAddrDetails=N&pageNum=1"
    try:
        response = requests.get(url)
        response.raise_for_status()
        results = response.json().get("results", [])
        if results:
            lat = results[0]["LATITUDE"]
            long = results[0]["LONGITUDE"]
            return lat, long
        else:
            return None, None  # Return None if no results are found
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        print(f"Error with address {address}: {e}")
        return None, None

# Apply the function with tqdm for progress bar
address_to_geom = {}
for address in tqdm(df["street_name"].unique(), desc="Processing addresses", unit="address"):
    lat, long = get_geom(address)
    address_to_geom[address]=f"{lat},{long}"

df["coords"] = df["street_name"].map(address_to_geom)

df.to_csv("Section 1 Datasets/prices_with_loc.csv", index=False)

