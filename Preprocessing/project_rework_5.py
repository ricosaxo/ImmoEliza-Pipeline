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


# NUMERICAL DATA

def identify_numerical_columns(df, exclude_columns=None):
    """
    Identifies the numerical columns in the DataFrame, excluding specified columns, columns ending with '_encoded',
    and binary columns containing only 0 and 1.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - exclude_columns (list, optional): List of column names to exclude from the numerical columns. Default is None.
    
    Returns:
    - list: A list of numerical column names after applying exclusions.
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # Select columns with numeric data types (int64 or float64)
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Exclude specified columns, columns ending in '_encoded', and binary columns
    numerical_columns = [
        col for col in numerical_columns
        if col not in exclude_columns 
        and not col.endswith('_encoded') 
        and not (df[col].nunique() == 2 and set(df[col].unique()) == {0, 1})
    ]
    
    return numerical_columns

def analyze_numerical_columns(df, exclude_columns=None, plot=True):
    """
    Analyzes the numerical columns in the DataFrame, with options to exclude specific columns,
    skip columns ending in '_encoded', and ignore binary columns (only 0 and 1 values).
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - exclude_columns (list, optional): List of column names to exclude from the numerical columns. Default is None.
    - plot (bool): If True, the function will plot boxplots for each numerical column. Default is True.
    
    Returns:
    - None: Prints summary statistics, skewness, and visualizes numerical columns.
    """
    # Identify numerical columns with additional exclusions
    numerical_columns = identify_numerical_columns(df, exclude_columns=exclude_columns)
    
    # Print summary statistics
    print("\nSummary Statistics for Numerical Columns:")
    print(df[numerical_columns].describe())
    
    # Detect outliers and calculate skewness
    for col in numerical_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        
        # Calculate skewness
        skewness = df[col].skew()
        
        print(f"\nColumn '{col}':")
        print(f"  Skewness: {skewness:.2f}")
        print(f"  Number of outliers: {outliers_count}")
        
        # Plot boxplot if enabled
        if plot:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[col], color='skyblue')
            plt.title(f"Boxplot of {col}")
            plt.xlabel(col)
            plt.tight_layout()
            plt.show()

def fill_missing_with_stat(df, columns=None, method='mode'):
    """
    Fills missing values in selected columns with either the mode, median, or mean (average)
    of that column in the dataframe.

    Parameters:
    - df (pd.DataFrame): The dataframe with columns to be filled.
    - columns (list of str): List of columns to fill. Default is None, which means all columns will be considered.
    - method (str): The method to use for filling missing values. Options are 'mode', 'median', or 'mean'. Default is 'mode'.
    
    Returns:
    - pd.DataFrame: The dataframe with missing values in selected columns replaced by the chosen method.
    """
    # Validate the 'method' argument
    if method not in ['mode', 'median', 'mean']:
        raise ValueError("Method must be one of 'mode', 'median', or 'mean'.")

    # If no columns specified, consider all columns in the dataframe
    if columns is None:
        columns = df.columns.tolist()

    # Loop through each selected column
    for column in columns:
        # Check if the column exists in the dataframe
        if column in df.columns:
            # Count missing values before filling
            missing_count = df[column].isnull().sum()

            if missing_count > 0:  # Only proceed if there are missing values
                # Fill based on the selected method
                if method == 'mode':
                    fill_value = df[column].mode()[0]  # Most frequent value
                elif method == 'median':
                    fill_value = df[column].median()  # Median value
                elif method == 'mean':
                    fill_value = df[column].mean()  # Mean (average) value

                # Fill missing values in the column
                df[column] = df[column].fillna(fill_value)
                
                # Print the details
                print(f"Column '{column}' had {missing_count} missing values. "
                      f"These have been filled with the {method}: {fill_value}.")

    return df


def main():
    
    #file path
    file_path = r'output\my_data_step4.csv'
    
    #load the data
    data = load_dataframe(file_path = file_path)
    
    # Fill missing values
    data = fill_missing_with_stat(data, columns=['Number_of_facades'])
    data = fill_missing_with_stat(data, columns=['landSurface', 'Living_area'], method='median')
    
    #Ensuring postal code is an object
    data['Postal_code'] = data['Postal_code'].astype(str)
    
    # Analyze the numerical columns in the DataFrame
    analyze_numerical_columns(data, exclude_columns= ['id'], plot=False)
    
    # Clipping for numerical columns 
    data['Price'] = data['Price'].clip(
        lower=data['Price'].quantile(0.05), 
        upper=data['Price'].quantile(0.95)
    )
    data['Living_area'] = data['Living_area'].clip(
        lower=data['Living_area'].quantile(0.1), 
        upper=data['Living_area'].quantile(0.90)
    )
    data['gardenSurface'] = data['gardenSurface'].clip(
        lower=data['gardenSurface'].quantile(0.1), 
        upper=data['gardenSurface'].quantile(0.90)
    )
    data['landSurface'] = data['landSurface'].clip(
        lower=data['landSurface'].quantile(0.1), 
        upper=data['landSurface'].quantile(0.90)
    )

    # Capping for 'Number_of_bedrooms'
    data['Number_of_bedrooms'] = data['Number_of_bedrooms'].clip(
        lower=data['Number_of_bedrooms'].quantile(0.01), 
        upper=data['Number_of_bedrooms'].quantile(0.99)
    )

    # Remove extreme latitude and longitude values but skip scaling
    data = data[(data['latitude'].between(-90, 90)) & (data['longitude'].between(-180, 180))]

    # Analyze the numerical columns in the DataFrame
    analyze_numerical_columns(data, exclude_columns= ['id', 'Postal_code'], plot=False)
    
    # Print summary of dataframe
    print_dataframe_summary(data)
    
    # Save the modified dataframe to CSV
    save_dataframe(data, 'output/my_data_step5')
    
if __name__ == "__main__":
    main()