"""
Data loading, cleaning, validation, and splitting module.
"""
import os
import yaml
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_data(file_path: str) -> pd.DataFrame:
    """
    Reads dataset from CSV file.
    
    Args:
        file_path: Path to raw CSV file.
        
    Returns:
        pd.DataFrame containing the dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file does not exist at path: {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Dataset at {file_path} is empty.")
    return df


def preprocess_data(df: pd.DataFrame, feature_columns: List[str] = None, target_column: str = None) -> pd.DataFrame:
    """
    Cleans raw dataframe: removes duplicates, handles missing values,
    and filters invalid records.
    
    Args:
        df: Input DataFrame.
        feature_columns: Expected feature column names.
        target_column: Target column name.
        
    Returns:
        Cleaned pd.DataFrame.
    """
    cleaned_df = df.copy()
    
    # 1. Remove duplicate records
    cleaned_df = cleaned_df.drop_duplicates()
    
    # 2. Handle missing values (impute numeric columns with median if any)
    numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if cleaned_df[col].isnull().sum() > 0:
            median_val = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_val)
            
    # 3. Filter invalid records (e.g. non-positive square feet or negative bedrooms)
    if "square_feet" in cleaned_df.columns:
        cleaned_df = cleaned_df[cleaned_df["square_feet"] > 0]
    if "bedrooms" in cleaned_df.columns:
        cleaned_df = cleaned_df[cleaned_df["bedrooms"] >= 0]
    if "bathrooms" in cleaned_df.columns:
        cleaned_df = cleaned_df[cleaned_df["bathrooms"] >= 0]
    if "location_score" in cleaned_df.columns:
        cleaned_df = cleaned_df[(cleaned_df["location_score"] >= 0) & (cleaned_df["location_score"] <= 10)]
    if target_column and target_column in cleaned_df.columns:
        cleaned_df = cleaned_df[cleaned_df[target_column] > 0]
        
    return cleaned_df


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits dataset into train and test feature/target sets.
    
    Args:
        df: Input DataFrame.
        target_column: Name of target column.
        test_size: Proportion of dataset to include in test split.
        random_state: Seed for reproducibility.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataframe.")
        
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def save_processed_data(df: pd.DataFrame, output_path: str):
    """Saves processed data to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to: {output_path}")


if __name__ == "__main__":
    cfg = load_config()
    raw_path = cfg["data"]["raw_path"]
    processed_path = cfg["data"]["processed_path"]
    target_col = cfg["data"]["target_column"]
    
    print(f"Loading raw data from: {raw_path}")
    raw_df = load_data(raw_path)
    print(f"Raw shape: {raw_df.shape}")
    
    cleaned = preprocess_data(raw_df, target_column=target_col)
    print(f"Cleaned shape: {cleaned.shape}")
    save_processed_data(cleaned, processed_path)

