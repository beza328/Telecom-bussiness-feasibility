import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # dtype of missing values
    mis_val_dtype = df.dtypes

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Dtype'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# def format_float(value):
#     return f'{value:,.2f}'


# def find_agg(df: pd.DataFrame, agg_column: str, agg_metric: str, col_name: str, top: int, order=False) -> pd.DataFrame:
#     new_df = df.groupby(agg_column)[agg_column].agg(agg_metric).reset_index(name=col_name). \
#         sort_values(by=col_name, ascending=order)[:top]
#     return new_df


def convert_bytes_to_megabytes(df, bytes_data):
    megabyte = 1 * 10e+5
    df[bytes_data] = df[bytes_data] / megabyte
    return df[bytes_data]


def fix_outlier(df, column):
    df[column] = np.where(df[column] > df[column].quantile(0.95), df[column].median(), df[column])
    return df[column]

def remove_outliers(df, column_to_process, z_threshold=3):
    # Apply outlier removal to the specified column
    z_scores = zscore(df[column_to_process])
    outlier_column = column_to_process + '_Outlier'
    df[outlier_column] = (np.abs(z_scores) > z_threshold).astype(int)
    df = df[df[outlier_column] == 0]  # Keep rows without outliers

    # Drop the outlier column as it's no longer needed
    df = df.drop(columns=[outlier_column], errors='ignore')

    return df


def remove_rows_with_missing_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Remove rows from the DataFrame where any of the specified columns have missing values.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): A list of column names to check for missing values.

    Returns:
    pd.DataFrame: A DataFrame with rows removed where any of the specified columns have missing values.
    """
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame")
    
    # Remove rows with missing values in any of the specified columns
    df_cleaned = df.dropna(subset=columns)
    
    return df_cleaned


    
def fill_missing_values(df: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """
    Fill missing values in the DataFrame for multiple columns using specified strategies.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (dict): A dictionary where keys are column names and values are the strategies:
                    ('mean', 'median', 'mode', 'interpolation', 'knn').

    Returns:
    pd.DataFrame: A DataFrame with missing values filled according to the specified strategies.
    """
    
    for column, strategy in columns.items():
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame")
        
        if strategy == 'mean':
            df[column].fillna(df[column].mean(), inplace=True)
        elif strategy == 'median':
            df[column].fillna(df[column].median(), inplace=True)
        elif strategy == 'mode':
            df[column].fillna(df[column].mode()[0], inplace=True)
        elif strategy == 'interpolation':
            df[column].interpolate(method='linear', inplace=True)
        elif strategy == 'knn':
            df = knn_impute(df, column)
        else:
            raise ValueError(f"Invalid strategy '{strategy}' for column '{column}'. Use 'mean', 'median', 'mode', 'interpolation', or 'knn'.")
    
    return df

def knn_impute(df: pd.DataFrame, column: str, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Fill missing values in the specified column using K-Nearest Neighbors (KNN) imputation.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to apply KNN imputation.
    n_neighbors (int): Number of neighbors to use for KNN. Default is 5.

    Returns:
    pd.DataFrame: A DataFrame with missing values in the specified column filled using KNN.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    # Apply KNN imputation to the entire DataFrame (to handle the specified column)
    df[column] = imputer.fit_transform(df[[column]])
    
    return df