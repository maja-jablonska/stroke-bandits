from typing import Dict, List
import pandas as pd
import click
import os
from sklearn.impute import KNNImputer

from scripts.encode import (encode_brainsite, encode_deathcode, encode_findiag, encode_gender, encode_haemtype, encode_infarct,
                            encode_stroketype, encode_yn, encode_yndq, encode_GCSEYE,
                            encode_GCSMOTOR, encode_GCSVERBAL)


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


def preprocess(df, columns: List[str], formats: Dict[str, str]):
    """
    Preprocess the input file.

    Args:
        df (pd.DataFrame): The dataframe to preprocess.
        columns (List[str]): The columns to preprocess.
        formats (Dict[str, str]): The formats of the columns.

    Returns:
        pd.DataFrame, pd.DataFrame: The preprocessed dataframe and the stats dataframe.
    """
    df = df.copy()
    df = df[columns]

    # You can add your preprocessing steps here, e.g.:
    numeric_cols = [col for col in columns if (formats[col] == 'float64')|(formats[col] == float)]
    
    # KNN Imputation for all columns
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = imputer.fit_transform(df[columns])
    df = pd.DataFrame(df_imputed, columns=df.columns)


    # Get means and stds for numeric columns (excluding 'deathcode')
    stats_df = df[numeric_cols].agg(['mean', 'std']).transpose()
    
    for col in numeric_cols:
        df = standardize(df, col)
        
    for col in columns:
        if formats[col] == 'YNDQ':
            df[col] = encode_yndq(df, col)
        elif formats[col] == 'gender':
            df[col] = encode_gender(df, col)
        elif formats[col] == 'deathcode':
            df[col] = encode_deathcode(df, col)
        elif formats[col] == 'STROKETYPE':
            df[col] = encode_stroketype(df, col)
        elif formats[col] == 'INFARCT':
            df[col] = encode_infarct(df, col)
        elif formats[col] == 'GCSEYE':
            df[col] = encode_GCSEYE(df, col)
        elif formats[col] == 'GCSMOTOR':
            df[col] = encode_GCSMOTOR(df, col)
        elif formats[col] == 'GCSVERBAL':
            df[col] = encode_GCSVERBAL(df, col)
        elif formats[col] == 'Y01N':
            df[col] = encode_yn(df, col)
        elif formats[col] == 'findiag':
            df[col] = encode_findiag(df, col)
        elif formats[col] == 'brainsite':
            df[col] = encode_brainsite(df, col)
        elif formats[col] == 'haemtype':
            df[col] = encode_haemtype(df, col)

    df = one_hot_encode(df)

    return df, stats_df


@click.command()
@click.option('--input-file', help='The input file to preprocess.')
@click.option('--output-file', help='The output file to save the preprocessed data to.')
def main(input_file, output_file):
    """
    Main function to preprocess the data.
    """
    df = pd.read_csv(input_file)
    df = preprocess(df)
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()