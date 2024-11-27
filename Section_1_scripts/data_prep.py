import pandas as pd
import os

files = os.listdir("Raw Datasets")

dataframes = [pd.read_csv(f"Raw Datasets/{file}")
              for file in files if "Resale" in file]

merged_df = pd.concat(dataframes, ignore_index=True)

cpi_df = pd.read_csv("Raw Datasets/sg_CPI_2019.csv", index_col='Year')

#Data prep for Remaining Lease column
def round_years(duration):
    if type(duration) == str and "month" in duration:
        year_month = duration.split("years")
        month = int(year_month[1].strip()[:2])/12
        year = int(year_month[0].strip())
        return year+month
    elif type(duration) == str and "year" in duration:
        return int(duration[:2])

# Adding in adjusted resale price based on CPI
def adjust_price(row: pd.DataFrame):
    txn_year = int(row['month'][:4])
    txn_price = row["resale_price"]
    current_cpi = cpi_df.loc[2024, "Annual"]
    txn_cpi = cpi_df.loc[txn_year, "Annual"]
    adjusted_price = txn_price * current_cpi/txn_cpi
    return adjusted_price


merged_df["remaining_lease"] = merged_df["remaining_lease"].apply(round_years)
merged_df["resale_price_adjusted"] = merged_df.apply(adjust_price, axis=1)

# Combining similar flat types to 1 category
merged_df['flat_type'] = merged_df['flat_type'].replace(
    'MULTI GENERATION', 'MULTI-GENERATION')
merged_df.to_csv("Section 1 Datasets/merged_prices.csv", index=False)
