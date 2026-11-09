"""
Script to generate a reproducible public synthetic housing dataset
matching real-world house price dynamics.
"""
import numpy as np
import pandas as pd
from pathlib import Path

def generate_housing_data(n_samples: int = 2500, random_seed: int = 42, output_path: str = "data/raw/housing.csv"):
    np.random.seed(random_seed)
    
    # Generate realistic housing features
    square_feet = np.random.normal(loc=2000, scale=600, size=n_samples).clip(600, 4800).round(1)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.05, 0.25, 0.45, 0.20, 0.05])
    bathrooms = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], size=n_samples, p=[0.10, 0.15, 0.35, 0.20, 0.12, 0.05, 0.03])
    location_score = np.random.uniform(1.0, 10.0, size=n_samples).round(2)
    year_built = np.random.randint(1960, 2024, size=n_samples)
    garage_spaces = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.15, 0.35, 0.40, 0.10])
    
    # Ground truth pricing function with noise
    base_price = 80000.0
    price = (
        base_price
        + 165.0 * square_feet
        + 18000.0 * bedrooms
        + 15000.0 * bathrooms
        + 28000.0 * location_score
        + 750.0 * (year_built - 1960)
        + 12000.0 * garage_spaces
        + np.random.normal(0, 15000, size=n_samples)
    ).round(2)
    
    df = pd.DataFrame({
        "square_feet": square_feet,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "location_score": location_score,
        "year_built": year_built,
        "garage_spaces": garage_spaces,
        "price": price
    })
    
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} housing records saved to {output_path}")
    print(df.head())
    print(df.describe())

if __name__ == "__main__":
    generate_housing_data()

