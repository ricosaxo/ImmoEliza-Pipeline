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

# Set options to show all columns
pd.set_option('display.max_columns', None)

def load_dataframe(file_path):
    """
    Opens a file (CSV or pickle) and loads it into a Pandas DataFrame.

    Parameters:
    file_path (str): Path to the CSV or pickle file.

    Returns:
    pd.DataFrame: DataFrame containing the data from the file, or None if loading fails.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            print("CSV file loaded successfully.")
        elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
            df = pd.read_pickle(file_path)
            print("Pickle file loaded successfully.")
        else:
            print("Error: Unsupported file format. Please use a .csv, .pkl, or .pickle file.")
            return None
        return df
    except FileNotFoundError:
        print("Error: The file was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except pd.errors.ParserError:
        print("Error: The file could not be parsed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def drop_column(df, column_name):
    if column_name in df.columns:
        df = df.drop(columns=[column_name])
        print(f"Dropped column: {column_name}")
    else:
        print(f"Column '{column_name}' not found in DataFrame.")
    return df

# Drop rows from the DataFrame based on specific conditions
def drop_rows_based_on_conditions(df, true_col=None, false_col=None, not_na_col=None, na_col=None):
    """
    Drops rows from the DataFrame based on specific conditions:
      - Removes rows where the value in `true_col` is True.
      - Removes rows where the value in `false_col` is False.
      - Removes rows where the value in `not_na_col` is not NaN.
      - Removes rows where the value in `na_col` is NaN.

    Parameters:
        df (pd.DataFrame): The DataFrame to filter.
        true_col (str, optional): Column where rows should be removed if the value is True.
        false_col (str, optional): Column where rows should be removed if the value is False.
        not_na_col (str, optional): Column where rows should be removed if the value is not NaN.
        na_col (str, optional): Column where rows should be removed if the value is NaN.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    # Drop rows where true_col is True
    if true_col:
        initial_row_count = df.shape[0]
        df = df[df[true_col] != True]
        rows_dropped = initial_row_count - df.shape[0]
        print(f"Rows dropped where '{true_col}' is True: {rows_dropped}")
    
    # Drop rows where false_col is False
    if false_col:
        initial_row_count = df.shape[0]
        df = df[df[false_col] != False]
        rows_dropped = initial_row_count - df.shape[0]
        print(f"Rows dropped where '{false_col}' is False: {rows_dropped}")
    
    # Drop rows where not_na_col is not NaN
    if not_na_col:
        initial_row_count = df.shape[0]
        df = df[df[not_na_col].isna()]
        rows_dropped = initial_row_count - df.shape[0]
        print(f"Rows dropped where '{not_na_col}' is not NaN: {rows_dropped}")
    
    # Drop rows where na_col is NaN
    if na_col:
        initial_row_count = df.shape[0]
        df = df[df[na_col].notna()]
        rows_dropped = initial_row_count - df.shape[0]
        print(f"Rows dropped where '{na_col}' is NaN: {rows_dropped}")
    
    return df

def replace_nan_with_false(df, columns):
    """
    Replaces NaN values in the specified columns with False and converts the column values to boolean.

    Parameters:
        df (pd.DataFrame): The DataFrame to modify.
        columns (list of str): List of column names where NaN values will be replaced with False.

    Returns:
        pd.DataFrame: The DataFrame with NaN values replaced and columns converted to boolean type.
    """
    for column in columns:
        if column in df.columns:
            df[column] = df[column].fillna(False).astype(bool)
            print(f"Replaced NaN in '{column}' with False")
        else:
            print(f"Column '{column}' not found in DataFrame.")
    
    return df

#Some additional changes

def edit_text_columns(data):
    """
    Edit specified columns in the dataframe by:
    - Capitalizing text (first letter uppercase, rest lowercase).
    - Replacing underscores with spaces.
    - Removing zip codes in parentheses from 'locality_name'.
    
    Parameters:
    - data (pd.DataFrame): The dataframe to modify.
    
    Returns:
    - pd.DataFrame: The dataframe with text modifications applied.
    """
    # Edit text in specified columns (replace underscores and capitalize text)
    text_edit_columns = ['Subtype', 'Kitchen_type', 'State_of_building']
    for column in text_edit_columns:
        data[column] = data[column].astype(str).str.replace('_', ' ').str.capitalize()

    # Edit text of cities and street names (capitalize each word)
    names_edit_columns = ['locality_name', 'street']
    for column in names_edit_columns:
        data[column] = data[column].astype(str).str.title()

    # Remove zip code from 'locality_name' (e.g., "Tielt (8700)" -> "Tielt")
    data['locality_name'] = data['locality_name'].str.replace(r"\s*\(\d+\)", "", regex=True)

    return data

#Postal_code check

def drop_invalid_values_by_column(data, column_name, length=4):
    """
    Drops rows where the specified column's values are not of the specified length,
    and prints how many rows were dropped.
    
    Parameters:
    - data (pd.DataFrame): The dataframe to clean.
    - column_name (str): The name of the column to check the length of its values.
    - length (int, optional): The desired length of the values in the column. Defaults to 4.
    
    Returns:
    - pd.DataFrame: The dataframe with invalid values removed.
    """
    # Count the number of rows before cleaning
    initial_row_count = data.shape[0]
    
    # Filter the dataframe based on the specified column and length
    data_cleaned = data[data[column_name].str.len() == length]
    
    # Calculate how many rows were dropped
    rows_dropped = initial_row_count - data_cleaned.shape[0]
    print(f"Rows dropped due to invalid '{column_name}' length (not {length} characters): {rows_dropped}")
    
    return data_cleaned


# Dropping rows that have no street or number assigned
def drop_rows_all_missing_columns(df, columns_to_check):
    """
    Drops rows from the dataframe where **all** of the specified columns have missing (NaN) values.
    
    Parameters:
    - df (pd.DataFrame): The dataframe to clean.
    - columns_to_check (list of str): List of column names to check for missing values.
    
    Returns:
    - pd.DataFrame: The cleaned dataframe with rows dropped where all specified columns have NaN values.
    """
    # Drop rows where all specified columns have missing values
    df_cleaned = df.dropna(subset=columns_to_check, how='all')
    
    # Print info after cleaning
    print(f"Rows dropped where all specified columns ({', '.join(columns_to_check)}) had missing values.")
    print("\nDataframe information after dropping rows:")
    df_cleaned.info()
    
    return df_cleaned

#Drops columns from the dataframe with a percentage of missing values

def clean_missing_data(data, threshold=0.3, exclude_columns=None):
    """
    Drops columns from the dataframe with a percentage of missing values 
    greater than the specified threshold, and prints details on the process.
    Optionally excludes specified columns from the drop check.
    
    Parameters:
    - data (pd.DataFrame): The dataframe to clean.
    - threshold (float): The maximum allowed percentage of missing values 
                         (between 0 and 1) for columns to be retained.
    - exclude_columns (list of str): Columns to exclude from the drop check. 
                                      Default is None, meaning no columns are excluded.
    
    Returns:
    - pd.DataFrame: The cleaned dataframe with columns exceeding the missing data
                    threshold removed (excluding specified columns).
    """
    
    # If exclude_columns is provided, ensure it's a list
    if exclude_columns is None:
        exclude_columns = []
    
    # Standardize missing values to np.nan (if 'Nan' or 'nan' as strings exist)
    data = data.replace(['Nan', 'nan'], np.nan)
    
    # Calculate the percentage of missing values per column
    missing_percent = data.isnull().mean()
    
    # Exclude the specified columns from the calculation
    missing_percent = missing_percent.drop(exclude_columns, errors='ignore')
    
    # Identify columns where the percentage of missing data is greater than the threshold
    columns_to_drop = missing_percent[missing_percent > threshold].index
    
    # Drop identified columns
    data = data.drop(columns=columns_to_drop)
    
    # Print dropped columns and their missing percentages
    print(f"Columns dropped due to exceeding missing value threshold of {threshold * 100}%:\n")
    for col in columns_to_drop:
        print(f"{col}: {missing_percent[col] * 100:.2f}% missing values")
    
    # Print the info of the cleaned dataframe
    print("\nDataframe information after dropping columns:")
    data.info()
    
    return data

# Convert the 'int64' and 'float64' columns to 'Int64' and 'float64' columns to nullable 'Int64'

def convert_columns(data, string_columns=None):
    """
    Convert columns of a DataFrame to appropriate types:
    - Boolean columns are converted to 0 and 1 (integers).
    - Numeric columns (int64 and float64) are converted to nullable integers (Int64).
    - Object columns are converted to categorical types, unless specified in `string_columns`, which are converted to strings.
    
    Parameters:
    - data (pd.DataFrame): The dataframe to be processed.
    - string_columns (list of str, optional): List of column names that should be kept as strings. 
      Defaults to None, meaning all object columns will be converted to categories except those specified.
    
    Returns:
    - pd.DataFrame: The dataframe with converted columns.
    
    Example:
    - convert_columns(data, string_columns=['locality_name', 'Postal_code'])
    """
    if string_columns is None:
        string_columns = []  # Default to an empty list if not provided

    # Convert boolean columns to integers (0 and 1)
    bool_columns = data.select_dtypes(include=['bool']).columns
    for column in bool_columns:
        data[column] = data[column].astype(int)

    # Convert int64 and float64 columns to nullable integers (Int64) where applicable
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    for column in numeric_columns:
        if column not in string_columns:
            # Use pd.to_numeric with 'coerce' to convert non-numeric values to NaN
            data[column] = pd.to_numeric(data[column], errors='coerce')
            # Convert to 'Int64' (nullable integer), NaN values will remain as NaN
            data[column] = data[column].astype('Int64', errors='ignore')

    # Convert object columns to categorical or string based on the `string_columns` list
    object_columns = data.select_dtypes(include=['object']).columns
    for column in object_columns:
        if column not in string_columns:
            data[column] = data[column].astype('category')  # Convert to category for better performance
        else:
            data[column] = data[column].astype('str')  # Explicitly convert to string

    # Provide summary of the dataframe after conversions
    print("\nDataframe information after converting columns:")
    data.info()

    return data

#Prints a summary of the DataFrame

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
    Saves a DataFrame as both a CSV and a pickle file in the specified directory.

    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - filename (str): The base name of the file without extension. Default is "data".
    - directory (str): Directory path where the files will be saved. Default is the current directory.
    - sep (str): The separator for the CSV file. Default is ','.
    - encoding (str): The encoding for the CSV file. Default is "utf-8".
    - index (bool): Whether to save the DataFrame index as a column. Default is False.

    Returns:
    - dict: Paths to the saved CSV and pickle files.
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Full paths for CSV and pickle files
    csv_path = os.path.join(directory, f"{filename}.csv")
    pickle_path = os.path.join(directory, f"{filename}.pkl")

    # Save the DataFrame to CSV
    df.to_csv(csv_path, sep=sep, encoding=encoding, index=index)
    print(f"DataFrame saved to CSV file at {csv_path}")

    # Save the DataFrame to pickle
    df.to_pickle(pickle_path)
    print(f"DataFrame saved to pickle file at {pickle_path}")

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
    data_cleaned = data_cleaned.drop_duplicates(subset=['street', 'number', 'Postal_code', 'latitude', 'longitude'], keep='first')

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

    # Print summary of the cleaned data
    print_dataframe_summary(data)

    # Save the cleaned data to a CSV file
    save_dataframe(data, filename="my_data_step1", directory="output")

if __name__ == "__main__":
    main()


