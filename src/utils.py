import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans



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



import pandas as pd

# Define a function to load the data
def load_data(filepath):
    return pd.read_csv(filepath)

# Function to identify the top N handsets
def get_top_n_handsets(df, handset_column, n=10):
    return df[handset_column].value_counts().head(n)

# Function to identify the top N manufacturers
def get_top_n_manufacturers(df, manufacturer_column, n=3):
    return df[manufacturer_column].value_counts().head(n)

# Function to get the top handsets for each manufacturer
def get_top_handsets_per_manufacturer(df, manufacturer_column, handset_column, top_n_manufacturers, top_handsets_per_manufacturer=5):
    top_handsets = {}
    for manufacturer in top_n_manufacturers.index:
        top_handsets[manufacturer] = df[df[manufacturer_column] == manufacturer][handset_column].value_counts().head(top_handsets_per_manufacturer)
    return top_handsets





def check_undefined(df):
    undefined_cols = {}
    for col in df.columns:
        undefined_count = (df[col] == 'undefined').sum()
        if undefined_count > 0:
            undefined_cols[col] = undefined_count
    return undefined_cols



# Function to calculate the number of XDR sessions
def calculate_xdr_sessions(df, session_id_column):
    return df[session_id_column].nunique()

# Function to calculate the session duration
def calculate_session_duration(df, start_time_column, end_time_column):
    df[start_time_column] = pd.to_datetime(df[start_time_column])
    df[end_time_column] = pd.to_datetime(df[end_time_column])
    df['Session_Duration'] = (df[end_time_column] - df[start_time_column]).dt.total_seconds()
    return df['Session_Duration']

# Function to calculate total download and upload data
def calculate_total_data(df, dl_column, ul_column):
    total_dl = df[dl_column].sum()
    total_ul = df[ul_column].sum()
    total_data_volume = total_dl + total_ul
    return total_dl, total_ul, total_data_volume

def calculate_total_volume_per_application(df, apps):
    """
    Calculate the total data volume (UL + DL) for each application.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the application UL and DL data.
    apps (list): A list of application names where each application has 'UL (bytes)' and 'DL (bytes)' columns.

    Returns:
    pandas.DataFrame: A DataFrame with new columns for the total data volume of each application.
    """
    for app in apps:
        ul_col = f'{app} UL (Bytes)'
        dl_col = f'{app} DL (Bytes)'
        total_col = f'{app} total (bytes)'
        
        # Calculate total volume (UL + DL) for the current application
        df[total_col] = df[ul_col] + df[dl_col]
    
    return df

def calculate_total_data_volume_across_apps(df, apps):
    """
    Calculate the total data volume for each application across all rows.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the total data columns per application.
    apps (list): A list of application names.

    Returns:
    pandas.DataFrame: A DataFrame with the total data volume for each application summed across all rows.
    """
    total_data_per_app = {}
    for app in apps:
        total_col = f'{app} total (bytes)'
        
        # Sum the total data volume for each application if the total column exists
        if total_col in df.columns:
            total_data_per_app[app] = df[total_col].sum()
    
    total_data_df = pd.DataFrame(list(total_data_per_app.items()), columns=['Application', 'Total_Volume'])
    return total_data_df    


def calculate_total_data_volume(df, apps):
    """
    Calculate the total data volume (DL + UL) for all applications combined.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the application UL and DL data.
    apps (list): A list of application names where each application has 'UL (bytes)' and 'DL (bytes)' columns.

    Returns:
    pandas.DataFrame: The DataFrame with a new column 'Total Data Volume' containing the combined DL + UL data.
    """
    df['Total Data Volume'] = 0
    for app in apps:
        ul_col = f'{app} UL (Bytes)'
        dl_col = f'{app} DL (Bytes)'
        
        if ul_col in df.columns and dl_col in df.columns:
            df['Total Data Volume'] += df[ul_col] + df[dl_col]
    
    return df

def correlation_analysis(df, apps):
    """
    Perform correlation analysis between each application's data and the total data volume.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the application data and total data volume.
    apps (list): A list of application names.

    Returns:
    pd.DataFrame: Correlation values between each application and the total data volume.
    """
    correlations = {}
    for app in apps:
        total_app_data = df[f'{app} UL (Bytes)'] + df[f'{app} DL (Bytes)']
        correlations[app] = df['Total Data Volume'].corr(total_app_data)
    
    return pd.DataFrame(correlations.items(), columns=['Application', 'Correlation with Total Data'])

def plot_relationship(df, apps):
    """
    Plot scatter plots and regression lines to visualize the relationship between each application and total data volume.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    apps (list): A list of application names.
    """
    for app in apps:
        total_app_data = df[f'{app} UL (Bytes)'] + df[f'{app} DL (Bytes)']
        sns.lmplot(x=total_app_data, y=df['Total Data Volume'])
        plt.xlabel(f'{app} Data (UL + DL)')
        plt.ylabel('Total Data Volume')
        plt.title(f'Relationship between {app} Data and Total Data Volume')
        plt.show()

def perform_regression_analysis(df, apps):
    """
    Perform regression analysis to model the relationship between each application's data and the total data volume.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    apps (list): A list of application names.

    Returns:
    dict: R-squared values for each application.
    """
    r2_values = {}
    for app in apps:
        total_app_data = df[f'{app} UL (Bytes)'] + df[f'{app} DL (Bytes)']
        X = total_app_data.values.reshape(-1, 1)
        y = df['Total Data Volume']
        
        # Perform linear regression
        reg_model = LinearRegression()
        reg_model.fit(X, y)
        y_pred = reg_model.predict(X)
        
        # Calculate R-squared value
        r2 = r2_score(y, y_pred)
        r2_values[app] = r2
    
    return r2_values

# Convert start and end times to datetime format
def convert_to_datetime(df, Start, End):
    df[Start] = pd.to_datetime(df[Start])
    df[End] = pd.to_datetime(df[End])
    return df

# Calculate session duration
def calculate_session_duration_from_times(df, Start, End):
    df['session_duration'] = (df[End] - df[Start]).dt.total_seconds() / 3600  # convert to hours
    return df.groupby('MSISDN/Number')['session_duration'].sum().reset_index(name='total_session_duration')

# Function to calculate session frequency per user
def calculate_session_frequency(df):
    return df.groupby('MSISDN/Number').size().reset_index(name='session_frequency')

# Function to calculate total traffic (DL + UL) per user
def calculate_total_traffic(df):
    df['total_traffic'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
    return df.groupby('MSISDN/Number')['total_traffic'].sum().reset_index(name='total_session_traffic')

# Main function to track user engagement metrics
def track_user_engagement(df, Start, End):
    df = convert_to_datetime(df, [Start, End])
    
    session_frequency = calculate_session_frequency(df)
    session_duration = calculate_session_duration_from_times(df, Start, End)
    total_traffic = calculate_total_traffic(df)
    
    # Merging the results into one dataframe
    engagement_df = session_frequency.merge(session_duration, on='MSISDN/Number')
    engagement_df = engagement_df.merge(total_traffic, on='MSISDN/Number')
    
    return engagement_df

def convert_to_datetime(df, columns):
    for col in columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')  # Coerce invalid parsing to NaT
    return df

# Get top N customers based on engagement metric
def get_top_customers(df, metric, top_n=10):
    return df[['MSISDN/Number', metric]].sort_values(by=metric, ascending=False).head(top_n)


def normalize_engagement_metrics(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

# Function to apply K-means clustering
def run_kmeans(df, columns, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[columns])
    return df, kmeans

# Function to plot the clusters
def plot_clusters(df, x_col, y_col):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], c=df['cluster'], cmap='viridis')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'K-means Clustering of {x_col} vs {y_col}')
    plt.colorbar(label='Cluster')
    plt.show()

# Function to normalize and cluster the engagement metrics
def classify_customers_kmeans(df, engagement_metrics, num_clusters=3):
    df_normalized = normalize_engagement_metrics(df, engagement_metrics)
    df_clustered, kmeans_model = run_kmeans(df_normalized, engagement_metrics, num_clusters)
    
    return df_clustered, kmeans_model




# Function to calculate the average values for numeric columns
def calculate_averages(df, numeric_cols, group_col='MSISDN/Number'):
    # Group by the 'MSISDN' column and calculate the mean for numeric columns
    avg_numeric = df.groupby(group_col)[numeric_cols].mean().reset_index()
    return avg_numeric

# Function to identify the most common handset type per customer
def most_common_handset_type(df, group_col='MSISDN/Number', handset_col='handset_type'):
    # Group by 'MSISDN' and find the mode (most frequent value) for the handset column
    most_common_handset = df.groupby(group_col)[handset_col].agg(lambda x: x.mode()[0]).reset_index()
    return most_common_handset

# Function to merge the average numeric data with handset type
def aggregate_customer_data(df, numeric_cols, handset_col='handset_type', group_col='MSISDN/Number'):
    # Calculate averages for numeric columns
    avg_numeric = calculate_averages(df, numeric_cols, group_col)

    # Get the most common handset type per customer
    common_handset = most_common_handset_type(df, group_col, handset_col)

    # Merge the two dataframes on 'MSISDN'
    result = pd.merge(avg_numeric, common_handset, on=group_col)
    return result

# Example usage
def main():
    # Columns of interest
    numeric_cols = ['TCP_retransmission', 'RTT', 'throughput']
    handset_col = 'handset_type'
    
    # Assuming df is your telecom dataset
    customer_data = aggregate_customer_data(df, numeric_cols, handset_col)

    # Display the results
    return customer_data

# Call the main function to process and display the data
if __name__ == "__main__":
    result = main()
    print(result.head())  # Print the first few rows of the result