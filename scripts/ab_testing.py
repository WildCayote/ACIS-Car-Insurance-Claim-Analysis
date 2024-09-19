import pandas as pd
import numpy as np
from scipy.stats import f_oneway

def annova_test(dependent_col: str, independent_col: str, data: pd.DataFrame):
    """
    Performs a one-way ANOVA test to determine if there are statistically significant differences 
    between the means of the dependent variable across the groups formed by the independent variable.
    
    Parameters:
    ----------
    dependent_col : str
        The name of the column containing the dependent variable (numeric) to test.
    independent_col : str
        The name of the column containing the independent variable (categorical) used to group the data.
    data : pd.DataFrame
        The DataFrame containing the dataset with both the dependent and independent variables.
    
    Returns:
    -------
    f_statistics : float
        The F-statistic value from the ANOVA test.
    p_value : float
        The p-value from the ANOVA test, which indicates the statistical significance.
    """

    # get the grouping along the independet_col
    data_groups = data.groupby(by=independent_col)

    # get the group names formed by using the independet_col
    group_names = list(data_groups.groups.keys())
    
    # get the groups and then put the dependent col values into a global list
    data_points = []
    for group_name in group_names:
        # group dataframe
        group_data = data_groups.get_group(name=group_name)

        # add the data points of the dependent column
        values = group_data[dependent_col]
        data_points.append(values)
    
    # perform an ANOVA test
    f_statistics, p_value = f_oneway(*data_points)

    return f_statistics, p_value

def test_hypothesis(null_hypothesis: str, p_value: np.float64):
    """
    Tests the given null hypothesis based on the p-value from the statistical test.
    
    Parameters:
    ----------
    null_hypothesis : str
        A description of the null hypothesis being tested.
    p_value : float
        The p-value from the statistical test used to determine whether to reject the null hypothesis.
    
    Returns:
    -------
    None
    
    Prints:
    -------
    Whether the null hypothesis is accepted or rejected based on the significance level (alpha = 0.05).
    """
    # if the p_value is less than 0.05 reject the null hypothesis
    if p_value < 0.05:
        print(f"Rejected the null hypothesis: {null_hypothesis} \np_value: {p_value}")
    else:
        print(f"Accepted the null hypothesis: {null_hypothesis} \np_value: {p_value}")