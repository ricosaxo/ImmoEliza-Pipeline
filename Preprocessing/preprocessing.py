import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from geopy.geocoders import Nominatim
import json
import os
import joblib
import time

def load_dataframe(file_path):
    """
    Load a CSV or pickle file into a DataFrame.
    """
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith(('.pkl', '.pickle')):
            return pd.read_pickle(file_path)
        else:
            raise ValueError("Unsupported file format. Use .csv, .pkl, or .pickle.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def drop_column(df, column_name):
    """Drop a column if it exists in the DataFrame."""
    if column_name in df.columns:
        return df.drop(columns=[column_name])
    print(f"Column '{column_name}' not found.")
    return df

def drop_rows_based_on_conditions(df, true_col=None, false_col=None, not_na_col=None, na_col=None):
    """Drop rows based on specified conditions."""
    if true_col:
        df = df[df[true_col] != True]
    if false_col:
        df = df[df[false_col] != False]
    if not_na_col:
        df = df[df[not_na_col].isna()]
    if na_col:
        df = df[df[na_col].notna()]
    return df

def replace_nan_with_false(df, columns):
    """Replace NaN values in specified columns with False and convert to boolean."""
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype(bool).fillna(False)
    return df

def edit_text_columns(df):
    """Capitalize and clean text columns."""
    text_columns = ['Subtype', 'Kitchen_type', 'State_of_building']
    name_columns = ['locality_name', 'street']

    for column in text_columns:
        if column in df.columns:
            df.loc[:, column] = df[column].astype(str).str.replace('_', ' ').str.capitalize()

    for column in name_columns:
        if column in df.columns:
            df.loc[:, column] = df[column].astype(str).str.title()

    if 'locality_name' in df.columns:
        df.loc[:, 'locality_name'] = df['locality_name'].str.replace(r"\s*\(\d+\)", "", regex=True)

    return df

def drop_invalid_values_by_column(df, column_name, length=4):
    """Drop rows where the specified column's values are not of the given length."""
    return df[df[column_name].str.len() == length]

def drop_rows_all_missing_columns(df, columns_to_check):
    """Drop rows where all specified columns have NaN values."""
    return df.dropna(subset=columns_to_check, how='all')

def clean_missing_data(df, threshold=0.3, exclude_columns=None):
    """Drop columns with missing values exceeding the threshold."""
    exclude_columns = exclude_columns or []
    df = df.replace(['Nan', 'nan'], np.nan).infer_objects(copy=False)
    missing_percent = df.isnull().mean()
    columns_to_drop = missing_percent[missing_percent > threshold].drop(exclude_columns, errors='ignore').index
    return df.drop(columns=columns_to_drop)

def convert_columns(df, string_columns=None):
    """Convert column types for better performance and usability."""
    string_columns = string_columns or []

    for column in df.select_dtypes(include=['bool']).columns:
        df[column] = df[column].astype(int)

    for column in df.select_dtypes(include=['int64', 'float64']).columns:
        if column not in string_columns:
            df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64', errors='ignore')

    for column in df.select_dtypes(include=['object']).columns:
        if column in string_columns:
            df[column] = df[column].astype(str)
        else:
            df[column] = df[column].astype('category')

    return df

def add_province_column(df, postal_code_column='Postal_code'):
    """
    Adds a 'Province' column to the DataFrame based on the postal code.
    """
    def get_province(postal_code):
        postal_code = str(postal_code).strip() if isinstance(postal_code, str) else postal_code
        try:
            postal_code = int(postal_code)
        except (ValueError, TypeError):
            return None

        province_ranges = {
            "Brussels": (1000, 1300),
            "Brabant_Walloon": (1300, 2000),
            "Antwerp": (2000, 3000),
            "Flemish Brabant": (3000, 3500),
            "Limburg": (3500, 4000),
            "Liège": (4000, 5000),
            "Namur": (5000, 6000),
            "Luxembourg": (6000, 7000),
            "Hainaut": (7000, 8000),
            "West Flanders": (8000, 9000),
            "East Flanders": (9000, 10000),
        }
        for province, (start, end) in province_ranges.items():
            if start <= postal_code < end:
                return province
        return None

    df['Province'] = df[postal_code_column].apply(get_province).astype('category')
    return df

def geocode_and_fill(df, cache_file='geocode_cache.json'):
    geolocator = Nominatim(user_agent="immo_eliza")

    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        print(f"Cache loaded from {cache_file}.")
    else:
        cache = {}
        print("No cache file found, starting with an empty cache.")

    def normalize_address(address):
        address = address.strip().lower()
        address = address.replace('str.', 'straat').replace('ave', 'avenue')
        address = address.replace('blvd', 'boulevard')
        address = ' '.join(address.split())
        return address

    def geocode_with_cache(address):
        normalized_address = normalize_address(address)

        if normalized_address in cache:
            print(f"Cache hit for address: {address}")
            return cache[normalized_address]

        try:
            location = geolocator.geocode(address, timeout=10)
            if location:
                result = (location.latitude, location.longitude)
            else:
                result = (None, None)
                print(f"No result found for address: {address}")
        except Exception as e:
            result = (None, None)
            print(f"Error geocoding {address}: {e}")

        if result != (None, None):
            cache[normalized_address] = result
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=4)
            print(f"Cache updated with address: {address}")

        time.sleep(1)
        return result

    missing_coords = df[df['latitude'].isna() | df['longitude'].isna()]
    total_missing = len(missing_coords)
    completed = 0
    failed = 0

    print(f"Starting geocoding for {total_missing} rows")

    for index, row in missing_coords.iterrows():
        address_1 = f"{row['street']} {row['number']}, {row['Postal_code']} {row['locality_name']}"
        address_2 = f"{row['street']} {row['number']}, {row['locality_name']}"
        address_3 = f"{row['locality_name']} {row['Postal_code']}"
        address_4 = f"{row['locality_name']}"

        addresses_to_try = [address_1, address_2, address_3, address_4]
        lat, lon = None, None

        for address in addresses_to_try:
            lat, lon = geocode_with_cache(address)
            if lat is not None and lon is not None:
                break

        df.at[index, 'latitude'] = lat
        df.at[index, 'longitude'] = lon

        if lat is not None and lon is not None:
            completed += 1
        else:
            failed += 1

        percent_complete = (completed / total_missing) * 100
        print(f"Completed {completed}/{total_missing} rows ({percent_complete:.2f}% complete)")

    print(f"Geocoding complete. {completed} rows successfully geocoded. {failed} rows failed.")
    return df

def assign_city_based_on_proximity_multiple_radii(df, cities_data, radius_list):
    cities_df = pd.DataFrame(cities_data)
    cities_gdf = gpd.GeoDataFrame(cities_df, geometry=gpd.points_from_xy(cities_df.Longitude, cities_df.Latitude))
    
    cities_gdf.set_crs(epsg=4326, allow_override=True, inplace=True)
    cities_gdf = cities_gdf.to_crs(epsg=3395)
    
    house_geo = pd.DataFrame(df[['id', 'latitude', 'longitude']])
    house_geo_gdf = gpd.GeoDataFrame(house_geo, geometry=gpd.points_from_xy(house_geo.longitude, house_geo.latitude))
    
    house_geo_gdf.set_crs(epsg=4326, allow_override=True, inplace=True)
    house_geo_gdf = house_geo_gdf.to_crs(epsg=3395)

    for radius in radius_list:
        cities_gdf['buffer'] = cities_gdf.geometry.buffer(radius * 1000)
        
        cities_gdf.set_geometry('buffer', inplace=True)
        
        joined_gdf = gpd.sjoin(house_geo_gdf, cities_gdf[['City', 'buffer']], how='left', predicate='intersects')
        
        joined_gdf = joined_gdf[joined_gdf['City'].notna()]
        
        joined_gdf = joined_gdf.drop_duplicates(subset='id', keep='first')
        
        house_geo_gdf[f'Assigned_City_{radius}'] = joined_gdf['City']
        
        df = pd.merge(df, house_geo_gdf[['id', f'Assigned_City_{radius}']], on='id', how='left')
        
        df[f'Has_Assigned_City_{radius}'] = df[f'Assigned_City_{radius}'].notna()
        df[f'Has_Assigned_City_{radius}'] = df[f'Has_Assigned_City_{radius}'].astype('bool')
        df[f'Has_Assigned_City_{radius}'] = df[f'Has_Assigned_City_{radius}'].astype(int)
    
    df = df.drop_duplicates(subset='id')
    
    print("\nDataframe information after adding assigning cities columns:")
    df.info()
    return df

def combine_subtypes(df, grouping_dict):
    if df is None or 'Subtype' not in df.columns:
        print("The DataFrame is None or does not contain a 'Subtype' column.")
        return None
    
    df_combined = df.copy()
    
    df_combined['Subtype'] = df_combined['Subtype'].astype(str)
    
    for new_subtype, original_subtypes in grouping_dict.items():
        df_combined['Subtype'] = df_combined['Subtype'].replace(original_subtypes, new_subtype)
    
    df_combined['Subtype'] = df_combined['Subtype'].astype('category')
    
    subtype_counts = df_combined['Subtype'].value_counts()
    subtype_percentages = (subtype_counts / len(df_combined)) * 100
    
    print("Value counts and percentages for each Subtype:")
    print("{:<25} {:<10} {}".format("Subtype", "Count", "Percentage"))
    print("-" * 45)
    for subtype, count in subtype_counts.items():
        print(f"{subtype:<25} {count:<10} {subtype_percentages[subtype]:.2f}")
    
    return df_combined


def filter_by_subtype(df, subtype):
    if df is None or 'Subtype' not in df.columns:
        print("The DataFrame is None or does not contain a 'Subtype' column.")
        return None
    
    return df[df['Subtype'] == subtype]

def analyze_categorical_data(df, threshold=0.05, exclude_columns=None):
    if df.empty:
        print("DataFrame is empty or invalid.")
        return
    exclude_columns = exclude_columns or []
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns.difference(exclude_columns)
    if categorical_cols.empty:
        print("No categorical columns found or all were excluded.")
        return

    for col in categorical_cols:
        print(f"\nAnalyzing column: {col}")
        value_counts = df[col].value_counts(normalize=True) * 100
        print(value_counts)
        rare_values = value_counts[value_counts < threshold * 100]
        if not rare_values.empty:
            print(f"\nRare values (<{threshold*100}%):\n{rare_values}")
        else:
            print(f"No rare values found for column {col}.")

def fill_missing_with_mode(df, include_columns=None, dtypes=None):
    dtypes = dtypes or ['category', 'object']
    include_columns = include_columns or df.select_dtypes(include=dtypes).columns
    for column in include_columns:
        if column in df.columns and df[column].dtype in dtypes:
            mode_value = df[column].mode()[0]
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                df[column] = df[column].fillna(mode_value)
                print(f"Column '{column}' had {missing_count} missing values filled with mode: '{mode_value}'.")
    return df

def target_encode(df, categorical_columns, target_column, drop_original=False, save_mappings=True):
    encoding_mappings = {}
    for col in categorical_columns:
        if col in df.columns:
            encoding = df.groupby(col)[target_column].mean()
            df[col + '_encoded'] = df[col].map(encoding)
            print(f"Target encoding applied to '{col}' based on '{target_column}'")
            encoding_mappings[col] = encoding.to_dict()
            if drop_original:
                df.drop(col, axis=1, inplace=True)
                print(f"Dropped original column '{col}'.")
    if save_mappings:
        for col, mapping in encoding_mappings.items():
            joblib.dump(mapping, f'output/{col}_encoding.pkl')
            print(f"Saved encoding mapping for '{col}' to 'output/{col}_encoding.pkl'")
    return df

def identify_numerical_columns(df, exclude_columns=None):
    exclude_columns = exclude_columns or []
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return [col for col in numerical_columns if col not in exclude_columns and not col.endswith('_encoded') 
            and not (df[col].nunique() == 2 and set(df[col].unique()) == {0, 1})]

def analyze_numerical_columns(df, exclude_columns=None, plot=True):
    numerical_columns = identify_numerical_columns(df, exclude_columns)
    
    print("\nSummary Statistics for Numerical Columns:")
    print(df[numerical_columns].describe())

    for col in numerical_columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        skewness = df[col].skew()
        
        print(f"\nColumn '{col}': Skewness: {skewness:.2f}, Outliers: {outliers_count}")
        
        if plot:
            sns.boxplot(x=df[col], color='skyblue').set(title=f"Boxplot of {col}", xlabel=col)
            plt.tight_layout()
            plt.show()

def fill_missing_with_stat(df, columns=None, method='mode'):
    if method not in ['mode', 'median', 'mean']:
        raise ValueError("Method must be one of 'mode', 'median', or 'mean'.")

    if columns is None:
        columns = df.columns.tolist()

    for column in columns:
        if column in df.columns:
            missing_count = df[column].isnull().sum()

            if missing_count > 0:
                if method == 'mode':
                    fill_value = df[column].mode()[0]
                elif method == 'median':
                    fill_value = df[column].median()
                elif method == 'mean':
                    fill_value = df[column].mean()

                df[column] = df[column].fillna(fill_value)
                
                print(f"Column '{column}' had {missing_count} missing values. "
                      f"These have been filled with the {method}: {fill_value}.")

    return df

def print_dataframe_summary(df):
    """
    Prints a summary of the DataFrame, including:
      - The first 20 rows of the DataFrame.
      - The count of NaN values in each column.
      - General information about the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to summarize.
    """
    print("First 15 rows of the DataFrame:")
    print(df.head(15))
    
    print("\nCount of NaN values in each column:")
    print(df.isna().sum())
    
    print("\nDataFrame Info:")
    print(df.info())

def save_dataframe(df, filename="data", directory=".", sep=",", encoding="utf-8", index=False):
    """
    Saves a DataFrame as CSV and pickle files in the specified directory.

    Returns:
        dict: Paths to the saved CSV and pickle files.
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Save DataFrame to CSV and pickle
    csv_path = os.path.join(directory, f"{filename}.csv")
    pickle_path = os.path.join(directory, f"{filename}.pkl")

    df.to_csv(csv_path, sep=sep, encoding=encoding, index=index)
    df.to_pickle(pickle_path)

    print(f"Data saved to:\nCSV: {csv_path}\nPickle: {pickle_path}")
    return {"csv": csv_path, "pickle": pickle_path}

def main():
    
    # File path for input data
    file_path = r'data\clean\immo_scraper_merged_with_landsurface.csv'

    # Load the data
    data = load_dataframe(file_path)

    # Drop unnecessary columns
    data = drop_column(data, 'Type_of_sale')

    # Drop rows based on specific conditions
    data = drop_rows_based_on_conditions(data, true_col='Starting_price', not_na_col='sale_annuity', na_col='Price')

    # Replace NaN with False in specific columns
    data = replace_nan_with_false(data, columns=['Swimming_Pool', 'hasTerrace', 'hasGarden', 'Furnished'])

    # Handle missing values in terraceSurface and gardenSurface
    missing_terraceSurface = data[data['hasTerrace'] == True]['terraceSurface'].isna().sum()
    total_terraceTrue = data[data['hasTerrace'] == True].shape[0]
    missing_terraceSurface = data[data['hasTerrace'] == True]['terraceSurface'].isna().sum()
    percentage_missing_terraceSurface = (missing_terraceSurface / total_terraceTrue) * 100
    print(f"Percentage of missing values in 'terraceSurface' where 'hasTerrace' is True: {percentage_missing_terraceSurface:.2f}%")

    missing_gardenSurface = data[data['hasGarden'] == True]['gardenSurface'].isna().sum()
    total_gardenTrue = data[data['hasGarden'] == True].shape[0]
    percentage_missing_gardenSurface = (missing_gardenSurface / total_gardenTrue) * 100
    print(f"Percentage of missing values in 'gardenSurface' where 'hasGarden' is True: {percentage_missing_gardenSurface:.2f}%")

    # Update gardenSurface to 0 where missing
    data['gardenSurface'] = data['gardenSurface'].fillna(0)

    # Remove duplicates based on 'id' and location info (street, number, postal code, latitude, longitude)
    data_cleaned = data.drop_duplicates(subset='id', keep='first')
    data = data_cleaned.drop_duplicates(subset=['street', 'number', 'Postal_code', 'latitude', 'longitude'], keep='first')

    # Edit text columns (capitalizing and removing zip codes in locality names)
    data = edit_text_columns(data)

    # Drop invalid postal codes
    data = drop_invalid_values_by_column(data, column_name='Postal_code', length=4)

    # Drop rows where street, number, latitude, and longitude are missing
    columns_to_check = ['street', 'number', 'longitude', 'latitude']
    data = drop_rows_all_missing_columns(data, columns_to_check)

    # Clean missing data by threshold (e.g., drop columns with more than 30% missing values)
    data = clean_missing_data(data, threshold=0.3)

    # Convert column types
    string_columns = ['locality_name', 'Postal_code', 'street', 'number']
    data = convert_columns(data, string_columns=string_columns)
    
    # Add province column
    data = add_province_column(data)
    
    # Fill missing geocoding information
    data = geocode_and_fill(data)
    
    # Assign cities based on proximity
    cities_data = {
        'City': ['Brussels', 'Antwerp', 'Ghent', 'Bruges', 'Liège', 'Namur', 'Leuven', 'Mons', 'Aalst', 'Sint-Niklaas'],
        'Latitude': [50.8503, 51.2194, 51.0543, 51.2093, 50.6293, 50.4811, 50.8794, 50.4542, 50.9402, 51.2170],
        'Longitude': [4.3517, 4.4025, 3.7174, 3.2247, 5.3345, 4.8708, 4.7004, 3.9460, 4.0710, 4.4155]
    }
    
    # Define radii in kilometers (e.g., 5 km, 10 km, 15 km)
    radius_list = [5, 10, 15]
    
    # Assign cities to the dataframe based on proximity
    data = assign_city_based_on_proximity_multiple_radii(data, cities_data, radius_list)
    
    #Convert the columns
    string_columns = ['locality_name', 'Postal_code', 'street', 'number']
    data = convert_columns(data, string_columns=string_columns)
    
    # Grouping the subtypes 
    grouping_dict = {
    'House': ['House', 'Town house', 'Country cottage'],
    'Luxury': ['Mansion', 'Castle', 'Manor house','Villa'],
    'Commercial': ['Mixed use building', 'Apartment block'],
    'Other': ['Bungalow', 'Farmhouse', 'Chalet', 'Exceptional property', 'Other property']
    }
    data = combine_subtypes(data, grouping_dict)
    
    #Filter the subtype ('House')
    subtype = 'House'
    data = filter_by_subtype(data, subtype)
    drop_column(data, column_name='Subtype')
    
    # Drop columns from the dataframe with a percentage of missing values
    data = clean_missing_data(data, threshold=0.3, exclude_columns=None)
    
    # Analyze categorical columns in the DataFrame
    analyze_categorical_data(data, threshold=0.05, exclude_columns=['street', 'number', 'locality_name', 'Postal_code'])
    
    #Assign the rare values to other values
    category_map_building = {'To restore': 'To renovate', 'To be done up': 'To renovate', 'Just renovated' : 'Good' }
    print("\nOriginal 'State_of_building' value counts:")
    print(data['State_of_building'].value_counts())
    data['State_of_building'] = data['State_of_building'].map(category_map_building).fillna(data['State_of_building'])
    print("\nUpdated 'State_of_building' value counts after mapping:")
    print(data['State_of_building'].value_counts())
    category_map_epc = {'A+': 'A', 'A++': 'A', 'G': 'F'}
    print("\nOriginal 'epc' value counts:")
    print(data['epc'].value_counts())
    data['epc'] = data['epc'].map(category_map_epc).fillna(data['epc'])
    print("\nUpdated 'epc' value counts after mapping:")
    print(data['epc'].value_counts())
    
    #Fill the missing values in specified columns with the mode (most frequent value) of that column in the dataframe
    data = fill_missing_with_mode(data, include_columns= ['State_of_building', 'epc'])
    
    #Perform Target Encoding on specified categorical columns
    data = target_encode(data, categorical_columns=['State_of_building', 'epc'], target_column='Price')
    
    # Fill missing values
    data = fill_missing_with_stat(data, columns=['Number_of_facades'])
    data = fill_missing_with_stat(data, columns=['landSurface', 'Living_area'], method='median')
    
    #Ensuring postal code is an object
    data['Postal_code'] = data['Postal_code'].astype(str)
    
    # Analyze the numerical columns in the DataFrame
    analyze_numerical_columns(data, exclude_columns= ['id'], plot=False)
    
    # Ensure that 'Price' and other columns are of type float before applying clip
    data['Price'] = data['Price'].astype(float).clip(
        lower=data['Price'].quantile(0.05),
        upper=data['Price'].quantile(0.95)
    )
    
    data['Living_area'] = data['Living_area'].astype(float).clip(
        lower=data['Living_area'].quantile(0.1),
        upper=data['Living_area'].quantile(0.90)
    )
    
    data['gardenSurface'] = data['gardenSurface'].astype(float).clip(
        lower=data['gardenSurface'].quantile(0.1),
        upper=data['gardenSurface'].quantile(0.90)
    )
    
    data['landSurface'] = data['landSurface'].astype(float).clip(
        lower=data['landSurface'].quantile(0.1),
        upper=data['landSurface'].quantile(0.90)
    )

    # Capping for 'Number_of_bedrooms'
    data['Number_of_bedrooms'] = data['Number_of_bedrooms'].astype(float).clip(
        lower=data['Number_of_bedrooms'].quantile(0.01),
        upper=data['Number_of_bedrooms'].quantile(0.99)
    )
    
    # Remove extreme latitude and longitude values but skip scaling
    data = data[(data['latitude'].between(-90, 90)) & (data['longitude'].between(-180, 180))]

    # Analyze the numerical columns in the DataFrame
    analyze_numerical_columns(data, exclude_columns= ['id', 'Postal_code'], plot=False)
    
    # Print summary of the cleaned data
    print_dataframe_summary(data)

    # Save the cleaned data to a CSV and pickle file
    save_dataframe(data, filename="output_preprocessing", directory="output")
    
if __name__ == "__main__":
    main()
