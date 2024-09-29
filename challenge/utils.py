import pandas as pd

def format_features_as_key_value(features: pd.DataFrame) -> str:
    """
    Format a DataFrame into a string with each column-value pair on a separate line.

    Args:
        features (pd.DataFrame): The DataFrame to format.

    Returns:
        str: Formatted string with each feature as 'column: value'.
    """
    formatted_features = []
    for index, row in features.iterrows():
        formatted_features.append(f"Feature set {index + 1}:")
        for col, val in row.items():
            formatted_features.append(f"  {col}: {val}")
        formatted_features.append("")  # Add a blank line between feature sets
    return "\n".join(formatted_features)
