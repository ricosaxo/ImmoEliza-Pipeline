from project_rework_1 import load_dataframe, clean_missing_data, drop_column, print_dataframe_summary, save_dataframe
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
import joblib


#CATEGORICAL DATA

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

#Analyzing categorical columns in the DataFrame

def analyze_categorical_data(df, threshold=0.05, exclude_columns=None):
    """
    Analyzes categorical columns in the DataFrame:
    - Prints the value counts for each categorical column.
    - Prints rare values (those that have a percentage below the given threshold).
    - Optionally excludes certain columns from the analysis.

    Parameters:
    - df: pandas DataFrame to analyze.
    - threshold: float, the percentage threshold below which values are considered rare. Default is 5% (0.05).
    - exclude_columns: list of column names to exclude from the analysis. Default is None (analyze all columns).
    """
    # Ensure DataFrame is valid
    if df is None or df.empty:
        print("The DataFrame is empty or invalid.")
        return

    # If no columns are provided to exclude, use an empty list
    if exclude_columns is None:
        exclude_columns = []

    # Select only categorical columns
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns

    # Remove excluded columns from the list
    categorical_cols = [col for col in categorical_cols if col not in exclude_columns]

    # Check if there are categorical columns left to analyze
    if not categorical_cols:
        print("No categorical columns found or all categorical columns were excluded.")
        return

    # Iterate over each categorical column
    for col in categorical_cols:
        print(f"\nAnalyzing column: {col}")
        
        # Get value counts and percentage
        value_counts = df[col].value_counts(normalize=True) * 100
        
        # Print value counts
        print("Value counts (%):")
        print(value_counts)
        
        # Print rare values based on the threshold
        rare_values = value_counts[value_counts < threshold * 100]
        
        if not rare_values.empty:
            print(f"\nRare values (less than {threshold * 100}%):")
            print(rare_values)
        else:
            print(f"\nNo rare values found for column {col} (less than {threshold * 100}%).")

#Fill missing values with mode

def fill_missing_with_mode(df, include_columns=None, dtypes=None):
    """
    Fills missing values in specified columns with the mode (most frequent value) 
    of that column in the dataframe and prints the column and count of filled values.
    
    Parameters:
    - df (pd.DataFrame): The dataframe with columns to be filled.
    - include_columns (list of str): Columns to include for filling. Default is None, which means all columns of specified dtypes are considered.
    - dtypes (list of str): List of dtypes to select columns for filling. Default is None.
    
    Returns:
    - pd.DataFrame: The dataframe with missing values in selected columns replaced by the mode.
    """
    # If no dtypes specified, use 'category' and 'object' types by default
    if dtypes is None:
        dtypes = ['category', 'object']
    
    # If include_columns is provided, ensure it's a list
    if include_columns is None:
        include_columns = df.select_dtypes(include=dtypes).columns.tolist()  # Use all columns of the given dtypes if not specified
    
    # Loop through each column in the dataframe
    for column in include_columns:
        # Check if the column exists in the dataframe and has the specified dtype
        if column in df.columns and df[column].dtype in dtypes:
            # Find the mode (most frequent value) of the column
            mode_value = df[column].mode()[0]
            
            # Count the number of missing values before filling
            missing_count = df[column].isnull().sum()
            
            if missing_count > 0:  # Only proceed if there are missing values
                # Fill missing values in the column with the mode
                df[column] = df[column].fillna(mode_value)
                
                # Print the details
                print(f"Column '{column}' had {missing_count} missing values. "
                      f"These have been filled with the mode: '{mode_value}'.")

    return df

def target_encode(df, categorical_columns, target_column, drop_original=False, save_mappings=True):
    """
    Applies target encoding to specified categorical columns by mapping each category to the mean of the target variable.
    Optionally saves the encoding mappings to separate .pkl files for each column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to encode.
    - categorical_columns (list of str): List of categorical columns to encode.
    - target_column (str): The target column used to compute the mean for each category.
    - drop_original (bool, optional): If True, removes the original categorical columns after encoding. Default is False.
    - save_mappings (bool, optional): If True, saves the encoding mappings for each column to individual .pkl files. Default is True.

    Returns:
    - pd.DataFrame: The modified DataFrame with encoded columns added and, optionally, the original categorical columns dropped.

    Side Effects:
    - Saves encoding mappings to .pkl files in the 'output' directory if `save_mappings` is True. The file names follow the format:
      'output/{column_name}_encoding.pkl'.

    Example Usage:
    --------------
    To target encode the 'State_of_building' and 'epc' columns based on the 'Price' column and save the mappings:
    
        df = target_encode(df, categorical_columns=['State_of_building', 'epc'], target_column='Price')

    To also drop the original columns after encoding:
    
        df = target_encode(df, categorical_columns=['State_of_building', 'epc'], target_column='Price', drop_original=True)

    Notes:
    - This function is intended for supervised encoding where each category in a feature is replaced by the average of the target variable 
      for that category.
    - The encoded column names are created by appending '_encoded' to the original column names.
    - If `drop_original` is True, the original columns specified in `categorical_columns` are removed from the DataFrame.

    """
    encoding_mappings = {}  # Dictionary to store encoding mappings for each column
    
    # Iterate over each categorical column to encode
    for col in categorical_columns:
        if col in df.columns:
            # Compute the mean of the target variable for each category in the column
            encoding = df.groupby(col)[target_column].mean()
            
            # Map each category in the column to its mean target value and add as a new encoded column
            df[col + '_encoded'] = df[col].map(encoding)
            print(f"Target encoding applied to column '{col}' based on '{target_column}'")
            
            # Store the encoding mapping in the dictionary
            encoding_mappings[col] = encoding.to_dict()
            
            # Optionally drop the original column after encoding
            if drop_original:
                df.drop(col, axis=1, inplace=True)
                print(f"Dropped original column '{col}' from the DataFrame.")
    
    # Optionally save each encoding mapping as a .pkl file
    if save_mappings:
        for col, mapping in encoding_mappings.items():
            joblib.dump(mapping, f'output/{col}_encoding.pkl')
            print(f"Saved encoding mapping for '{col}' to 'output/{col}_encoding.pkl'")
    
    return df


def main():
    
    #file path
    file_path = r'output\my_data_step3.csv'
    
    #load the data
    data = load_dataframe(file_path = file_path)
    
    # Drop columns from the dataframe with a percentage of missing values
    data = clean_missing_data(data, threshold=0.3, exclude_columns=None)
    
    # Apply the function to drop rows where all specified columns are NaN
    # columns_to_check = ['street', 'number']
    # data = drop_rows_all_missing_columns(data, columns_to_check)
    
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
    
    # Print summary of dataframe
    print_dataframe_summary(data)
    
    # Save the modified dataframe to CSV
    save_dataframe(data, 'output/my_data_step4')
    
if __name__ == "__main__":
    main()