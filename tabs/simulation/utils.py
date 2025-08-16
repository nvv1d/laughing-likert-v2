import re
import pandas as pd
import numpy as np
from collections import defaultdict

def detect_scales_by_pattern(columns):
    """
    Automatically detect scales based on naming patterns like A1, A2, A3, B1, B2, B3, etc.
    Returns a dictionary with scale names as keys and lists of items as values.
    """
    scales = defaultdict(list)

    # Pattern 1: Letter + Number (A1, A2, B1, B2, etc.)
    pattern1 = re.compile(r'^([A-Za-z]+)(\d+)$')

    # Pattern 2: Word + Number (Scale1_1, Scale1_2, Scale2_1, etc.)
    pattern2 = re.compile(r'^([A-Za-z_]+?)(\d+)(?:_(\d+))?$')

    # Pattern 3: Common prefixes (att1, att2, sat1, sat2, etc.)
    pattern3 = re.compile(r'^([A-Za-z]+)(\d+)$')

    for col in columns:
        # Try pattern 1: A1, A2, B1, B2
        match1 = pattern1.match(col)
        if match1:
            prefix = match1.group(1).upper()
            number = int(match1.group(2))
            scales[f"Scale_{prefix}"].append((col, number))
            continue

        # Try pattern 2: Scale1_1, Scale1_2, Scale2_1
        match2 = pattern2.match(col)
        if match2:
            prefix = match2.group(1)
            if match2.group(3):  # Has underscore number
                scale_num = int(match2.group(2))
                item_num = int(match2.group(3))
                scales[f"Scale_{prefix}_{scale_num}"].append((col, item_num))
            else:
                number = int(match2.group(2))
                scales[f"Scale_{prefix}"].append((col, number))
            continue

    # Sort items within each scale by their numbers and return just the column names
    sorted_scales = {}
    for scale_name, items in scales.items():
        if len(items) >= 2:  # Only keep scales with at least 2 items
            sorted_items = sorted(items, key=lambda x: x[1])
            sorted_scales[scale_name] = [item[0] for item in sorted_items]

    return sorted_scales

def ensure_numeric_data(data, columns):
    """
    Ensure specified columns are numeric, handling errors gracefully.
    
    Parameters:
    - data: DataFrame to convert
    - columns: List of column names to convert
    
    Returns:
    - DataFrame with numeric columns
    """
    numeric_data = data.copy()
    for col in columns:
        if col in numeric_data.columns:
            numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
    return numeric_data

def safe_calculate_stats(data, stat_functions):
    """
    Safely calculate statistics with error handling.
    
    Parameters:
    - data: DataFrame to calculate stats on
    - stat_functions: Dictionary of {'stat_name': function}
    
    Returns:
    - Dictionary of successfully calculated statistics
    """
    results = {}
    for stat_name, func in stat_functions.items():
        try:
            results[stat_name] = func(data)
        except Exception as e:
            print(f"Warning: Could not calculate {stat_name}: {str(e)}")
    return results

def normalize_distribution(distribution_dict):
    """
    Normalize a distribution dictionary so values sum to 1.
    
    Parameters:
    - distribution_dict: Dictionary with values as probabilities
    
    Returns:
    - Normalized dictionary
    """
    total = sum(distribution_dict.values())
    if total > 0:
        return {k: v/total for k, v in distribution_dict.items()}
    return distribution_dict

def get_common_values(series1, series2):
    """
    Get common values between two pandas Series for distribution comparison.
    
    Parameters:
    - series1, series2: pandas Series
    
    Returns:
    - List of sorted common values
    """
    values1 = set(series1.dropna().unique())
    values2 = set(series2.dropna().unique())
    return sorted(values1 | values2)
