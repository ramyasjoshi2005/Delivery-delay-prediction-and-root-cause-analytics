import pandas as pd
import numpy as np

def engineer_features(input_path, output_path):
    print("Loading cleaned data...")
    df = pd.read_csv(input_path)
    
    # 1. Date Features
    df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
    df['order_month'] = df['order date (DateOrders)'].dt.month
    df['order_day_of_week'] = df['order date (DateOrders)'].dt.dayofweek
    df['is_weekend'] = df['order_day_of_week'].isin([5, 6]).astype(int)
    df['quarter'] = df['order date (DateOrders)'].dt.quarter
    
    # Peak season indicator (e.g., Q4 is typically peak for e-commerce)
    df['peak_season'] = (df['quarter'] == 4).astype(int)
    
    # Keep original date column for train/test split, we'll drop it in train.py
    # df = df.drop(columns=['order date (DateOrders)'])
    
    print("Features engineered.")
    print(df.columns.tolist())
    
    df.to_csv(output_path, index=False)
    print(f"Feature-engineered data saved to {output_path}. Shape: {df.shape}")

if __name__ == "__main__":
    engineer_features('data/processed/clean_data.csv', 'data/processed/features_data.csv')
