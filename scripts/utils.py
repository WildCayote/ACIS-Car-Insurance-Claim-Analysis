import pandas as pd

def count_group_contribution(data: pd.DataFrame, grouping_col: str):
    """
    Prints the percentage of the total data each group (in a specified column) holds.
    
    Parameters:
    ----------
    group_col : str
        The name of the column to group by (e.g., 'Gender').
    data : pd.DataFrame
        The DataFrame containing the dataset.
    
    Returns:
    -------
    None
    """
    # Group the data by the specified column
    data_groups = data.groupby(by=grouping_col)
    
    # Get the total number of rows in the dataset
    total_count = len(data)
    
    # Iterate over each group and calculate its percentage
    for group_name, group_data in data_groups:
        group_count = len(group_data)
        percentage = (group_count / total_count) * 100
        print(f"{group_name}: {percentage:.2f}% of the data")