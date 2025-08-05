import pandas as pd
import click
import os


def one_hot_encode(df):
    # Identify categorical columns (object or category dtype)
    category_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return pd.get_dummies(df, columns=category_cols, drop_first=True)


def convert_to_bool(df, column_name):
    df_encoded = df.copy()
    df_encoded[column_name] = df_encoded[column_name].notna().astype(int)
    return df_encoded


def min_max_normalize(df, column_name):
    df_encoded = df.copy()
    age_min = df_encoded[column_name].min()
    age_max = df_encoded[column_name].max()
    df_encoded[column_name] = (df_encoded[column_name] - age_min) / (age_max - age_min)
    return df_encoded


def standardize(df, column_name):
    df_encoded = df.copy()
    df_encoded[column_name] = (df_encoded[column_name] - df_encoded[column_name].mean()) / df_encoded[column_name].std()
    return df_encoded


def preprocess(df):
    """
    Preprocess the input file.

    Args:
        df (pd.DataFrame): The dataframe to preprocess.

    Returns:
        pd.DataFrame, pd.DataFrame: The preprocessed dataframe and the stats dataframe.
    """
    df = convert_to_bool(df, 'deathcode')
    # You can add your preprocessing steps here, e.g.:
    numeric_cols = df.select_dtypes(include=['number']).columns.difference(['deathcode']).tolist()
    
    # Get means and stds for numeric columns (excluding 'deathcode')
    stats_df = df[numeric_cols].agg(['mean', 'std']).transpose()
    
    for col in numeric_cols:
        df = standardize(df, col)

    df = one_hot_encode(df)

    return df, stats_df
