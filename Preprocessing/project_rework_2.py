
from project_rework_1 import load_dataframe, convert_columns, print_dataframe_summary, save_dataframe
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
import json
import os
import pickle

def add_province_column(df, postal_code_column='Postal_code'):
    """
    Adds a 'Province' column to the DataFrame based on the 'Postal_code' column.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the 'Postal_code' column.
    postal_code_column (str): The name of the column containing postal codes (default is 'Postal_code').
    
    Returns:
    pd.DataFrame: DataFrame with the 'Province' column added.
    """
    def get_province(postal_code):
        """Returns the province based on the postal code."""
        if isinstance(postal_code, str):  # If postal code is a string, attempt to convert to int
            postal_code = postal_code.strip()  # Clean any extra spaces
            try:
                postal_code = int(postal_code)
            except ValueError:
                return None  # If conversion fails, return None (for invalid postal codes)
        
        if isinstance(postal_code, int):  # Only proceed if postal code is an integer
            if postal_code >= 1000 and postal_code < 2000:
                return 'Brussels' if postal_code < 1300 else 'Brabant_Walloon'
            elif postal_code >= 2000 and postal_code < 3000:
                return 'Antwerp'
            elif postal_code >= 4000 and postal_code < 5000:
                return 'Liège'
            elif postal_code >= 5000 and postal_code < 6000:
                return 'Namur'
            elif postal_code >= 6000 and postal_code < 7000:
                return 'Luxembourg'
            elif postal_code >= 7000 and postal_code < 8000:
                return 'Hainaut'
            elif postal_code >= 8000 and postal_code < 9000:
                return 'West Flanders'
            elif postal_code >= 9000 and postal_code < 10000:
                return 'East Flanders'
            elif postal_code >= 3000 and postal_code < 3500:
                return 'Flemish Brabant'
            elif postal_code >= 3500 and postal_code < 4000:
                return 'Limburg'
        return None  # For postal codes that do not match the known pattern

    # Ensure the 'Postal_code' is treated as an integer for logic
    df[postal_code_column] = pd.to_numeric(df[postal_code_column], errors='coerce')
    
    # Apply the 'get_province' function to the 'Postal_code' column to create the 'Province' column
    df['Province'] = df[postal_code_column].apply(get_province)
    
    # Assign the 'Province' column to category type for efficiency
    df['Province'] = df['Province'].astype('category')
    
    # Convert 'Postal_code' back to object type for efficiency
    df[postal_code_column] = df[postal_code_column].astype('object')
    
    print("\nDataframe information after adding column 'Province':")
    df.info()
    
    return df

#Filling Missing values in longitude and latitude based on address

def geocode_and_fill(df, cache_file='geocode_cache.json'):
    """
    Fills missing latitude and longitude in a DataFrame by geocoding addresses.
    Caches geocoding results to avoid redundant API calls. 
    Saves the cache to a specified file.
    
    Parameters:
    - df (pd.DataFrame): The dataframe containing the data to geocode.
    - cache_file (str): The file to store the cache of geocoded addresses.
    
    Returns:
    - pd.DataFrame: The dataframe with latitude and longitude filled.
    """
    # Initialize the geolocator
    geolocator = Nominatim(user_agent="immo_eliza")

    # Load cache from file if it exists
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        print(f"Cache loaded from {cache_file}.")
    else:
        cache = {}
        print("No cache file found, starting with an empty cache.")

    # Normalize address format for consistency
    def normalize_address(address):
        # Convert address to lowercase, remove excess spaces, and standardize common abbreviations
        address = address.strip().lower()
        address = address.replace('str.', 'straat').replace('ave', 'avenue')
        address = address.replace('blvd', 'boulevard')  # Add any other common abbreviations
        address = ' '.join(address.split())  # Remove extra spaces
        return address

    # Geocode an address and update the cache
    def geocode_with_cache(address):
        normalized_address = normalize_address(address)

        # Check if the normalized address is already in the cache
        if normalized_address in cache:
            print(f"Cache hit for address: {address}")
            return cache[normalized_address]

        # Geocode and add result to cache if not found
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

        # Store result in the cache if valid
        if result != (None, None):
            cache[normalized_address] = result
            # Save the updated cache to disk
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=4)
            print(f"Cache updated with address: {address}")

        time.sleep(1)  # Respect Nominatim rate limit
        return result

    # Find rows with missing latitude or longitude
    missing_coords = df[df['latitude'].isna() | df['longitude'].isna()]
    total_missing = len(missing_coords)
    completed = 0
    failed = 0

    print(f"Starting geocoding for {total_missing} rows")

    # Geocode missing addresses and update the DataFrame
    for index, row in missing_coords.iterrows():
        # Create different combinations of the address for better geocoding results
        address_1 = f"{row['street']} {row['number']}, {row['Postal_code']} {row['locality_name']}"
        address_2 = f"{row['street']} {row['number']}, {row['locality_name']}"
        address_3 = f"{row['locality_name']} {row['Postal_code']}"
        address_4 = f"{row['locality_name']}"

        # Try the addresses in order, to increase chances of finding a match
        addresses_to_try = [address_1, address_2, address_3, address_4]
        lat, lon = None, None

        for address in addresses_to_try:
            lat, lon = geocode_with_cache(address)
            if lat is not None and lon is not None:
                break  # Stop once a valid result is found

        # Update the DataFrame with the coordinates
        df.at[index, 'latitude'] = lat
        df.at[index, 'longitude'] = lon

        if lat is not None and lon is not None:
            completed += 1
        else:
            failed += 1

        percent_complete = (completed / total_missing) * 100
        print(f"Completed {completed}/{total_missing} rows ({percent_complete:.2f}% complete)")

    # Final summary
    print(f"Geocoding complete. {completed} rows successfully geocoded. {failed} rows failed.")
    return df

#A function to assign cities to each row in the dataframe based on proximity
def assign_city_based_on_proximity_multiple_radii(df, cities_data, radius_list):
    """
    This function assigns cities to each row in the dataframe based on proximity 
    to the 10 main Belgian cities within multiple given distance radii.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing house data with 'latitude' and 'longitude' columns.
    cities_data (dict): Dictionary containing city data ('City', 'Latitude', 'Longitude').
    radius_list (list): List of radii (in kilometers) to consider for proximity.
    
    Returns:
    df (pd.DataFrame): Updated DataFrame with 'Assigned_City' columns and proximity-based boolean columns.
    """
    # Convert cities data into GeoDataFrame
    cities_df = pd.DataFrame(cities_data)
    cities_gdf = gpd.GeoDataFrame(cities_df, geometry=gpd.points_from_xy(cities_df.Longitude, cities_df.Latitude))
    
    # Set CRS for cities GeoDataFrame to WGS84 (EPSG:4326)
    cities_gdf.set_crs(epsg=4326, allow_override=True, inplace=True)
    
    # Reproject cities to a projected CRS (e.g., EPSG:3395 or EPSG:3857 for meters)
    cities_gdf = cities_gdf.to_crs(epsg=3395)
    
    # Convert house data into GeoDataFrame
    house_geo = pd.DataFrame(df[['id', 'latitude', 'longitude']])
    house_geo_gdf = gpd.GeoDataFrame(house_geo, geometry=gpd.points_from_xy(house_geo.longitude, house_geo.latitude))
    
    # Set CRS for house GeoDataFrame to WGS84 (EPSG:4326)
    house_geo_gdf.set_crs(epsg=4326, allow_override=True, inplace=True)
    
    # Reproject house data to the same projected CRS as cities
    house_geo_gdf = house_geo_gdf.to_crs(epsg=3395)

    for radius in radius_list:
        # Create buffer around each city (in meters)
        cities_gdf['buffer'] = cities_gdf.geometry.buffer(radius * 1000)  # Convert km to meters
        
        # Set the 'buffer' as the active geometry column for cities
        cities_gdf.set_geometry('buffer', inplace=True)
        
        # Perform spatial join between houses and cities using the buffer as the area of interest
        joined_gdf = gpd.sjoin(house_geo_gdf, cities_gdf[['City', 'buffer']], how='left', predicate='intersects')
        
        # Drop rows where no city was assigned (i.e., no intersection)
        joined_gdf = joined_gdf[joined_gdf['City'].notna()]
        
        # Drop duplicates based on the 'id', keeping the first city that intersects with the house
        joined_gdf = joined_gdf.drop_duplicates(subset='id', keep='first')
        
        # Add the assigned city for this radius to the house GeoDataFrame
        house_geo_gdf[f'Assigned_City_{radius}'] = joined_gdf['City']
        
        # Merge the results back into the original dataframe
        df = pd.merge(df, house_geo_gdf[['id', f'Assigned_City_{radius}']], on='id', how='left')
        
        # Create boolean column indicating if a city was assigned within the radius
        df[f'Has_Assigned_City_{radius}'] = df[f'Assigned_City_{radius}'].notna()
        df[f'Has_Assigned_City_{radius}'] = df[f'Has_Assigned_City_{radius}'].astype('bool')
        df[f'Has_Assigned_City_{radius}'] = df[f'Has_Assigned_City_{radius}'].astype(int)
    
    # Optionally, drop any remaining duplicates if the merge operation introduced new ones
    df = df.drop_duplicates(subset='id')
    
    print("\nDataframe information after adding assigning cities columns:")
    df.info()
    return df

def main():
    # File path
    file_path = r'output\my_data_step1.csv'
    
    # Load the data
    data = load_dataframe(file_path=file_path)
    
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
    
    # Print summary of dataframe
    print_dataframe_summary(data)
    
    # Save the modified dataframe to CSV
    save_dataframe(data, 'output/my_data_step2')
    
if __name__ == "__main__":
    main()