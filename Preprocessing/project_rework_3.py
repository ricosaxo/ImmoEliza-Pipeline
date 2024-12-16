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

def plot_subtype_distribution(df):
    """
    Plots the distribution of property subtypes and displays their counts and percentages.

    Parameters:
    df (pd.DataFrame): The DataFrame containing a 'Subtype' column.

    Returns:
    None
    """
    # Check if 'Subtype' column exists in the DataFrame
    if 'Subtype' not in df.columns:
        print("The DataFrame does not contain a 'Subtype' column.")
        return
    
    # Get the value counts with percentages
    subtype_counts = df['Subtype'].value_counts()
    subtype_percentages = df['Subtype'].value_counts(normalize=True) * 100  # Get percentages
    
    # Combine counts and percentages into a DataFrame for easy display
    subtype_summary = pd.DataFrame({
        'Count': subtype_counts,
        'Percentage': subtype_percentages.round(2)  # Round percentages to 2 decimal places
    })
    
    # Display the summary table
    print("Value counts and percentages for each Subtype:")
    print(subtype_summary)
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    subtype_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Property Subtypes')
    plt.xlabel('Property Subtype')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def combine_subtypes(df, grouping_dict):
    """
    Combines property subtypes in the DataFrame based on the provided grouping dictionary.

    Parameters:
    df (pd.DataFrame): The DataFrame containing a 'Subtype' column.
    grouping_dict (dict): A dictionary where keys are the new subtypes and values are lists of old subtypes to be combined.

    Returns:
    pd.DataFrame: A new DataFrame with combined subtypes.
    """
    # Ensure DataFrame is valid and has a 'Subtype' column
    if df is None or 'Subtype' not in df.columns:
        print("The DataFrame is None or does not contain a 'Subtype' column.")
        return None
    
    # Make a copy to avoid modifying the original DataFrame
    df_combined = df.copy()
    
    # Convert 'Subtype' to string to allow flexible replacement without unique category constraints
    df_combined['Subtype'] = df_combined['Subtype'].astype(str)
    
    # Perform replacement based on the grouping dictionary
    for new_subtype, original_subtypes in grouping_dict.items():
        df_combined['Subtype'] = df_combined['Subtype'].replace(original_subtypes, new_subtype)
    
    # Convert 'Subtype' back to categorical
    df_combined['Subtype'] = df_combined['Subtype'].astype('category')
    
    # Calculate value counts and percentages for each subtype
    subtype_counts = df_combined['Subtype'].value_counts()
    subtype_percentages = (subtype_counts / len(df_combined)) * 100
    
    # Print counts and percentages
    print("Value counts and percentages for each Subtype:")
    print("{:<25} {:<10} {}".format("Subtype", "Count", "Percentage"))
    print("-" * 45)
    for subtype, count in subtype_counts.items():
        print(f"{subtype:<25} {count:<10} {subtype_percentages[subtype]:.2f}")
    
    return df_combined

def filter_by_subtype(df, subtype):
    """
    Filters the DataFrame to include only the rows where the 'Subtype' column matches the specified subtype.

    Parameters:
    df (pd.DataFrame): The DataFrame containing a 'Subtype' column.
    subtype (str): The value of the 'Subtype' column to filter by.

    Returns:
    pd.DataFrame: A DataFrame filtered by the specified subtype.
    """
    # Ensure DataFrame is valid and has a 'Subtype' column
    if df is None or 'Subtype' not in df.columns:
        print("The DataFrame is None or does not contain a 'Subtype' column.")
        return None
    
    # Filter the DataFrame for the specified subtype
    filtered_df = df[df['Subtype'] == subtype]
    
    return filtered_df

def main():
    
    #file path
    file_path = r'output\my_data_step2.csv'
    
    #load the data
    data = load_dataframe(file_path = file_path)
    
    #Visualize the data based on subtype
    plot_subtype_distribution(data)
    
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
    
    # Print summary of dataframe
    print_dataframe_summary(data)
    
    # Save the modified dataframe to CSV
    save_dataframe(data, 'output/my_data_step3')
    

if __name__ == "__main__":
    main()

