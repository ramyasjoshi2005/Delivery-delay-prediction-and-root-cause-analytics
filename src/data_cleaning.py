import pandas as pd
import numpy as np

def clean_data(input_path, output_path):
    print("Loading data...")
    df = pd.read_csv(input_path, encoding='latin-1')
    
    # 1. Define Prediction Point & Target
    # Target: Late = 1 if Actual Delivery Date > Promised Delivery Date
    # In this dataset: Days for shipping (real) > Days for shipment (scheduled)
    df['Late'] = (df['Days for shipping (real)'] > df['Days for shipment (scheduled)']).astype(int)
    
    # Check if this matches Late_delivery_risk exactly (sanity check)
    match_rate = (df['Late'] == df['Late_delivery_risk']).mean()
    print(f"Target matches Late_delivery_risk: {match_rate:.2%}")
    
    # 2. Select variables available at order placement
    # Drop leakage columns: 'Days for shipping (real)', 'Delivery Status', 'Late_delivery_risk', 'shipping date (DateOrders)', 'Order Status'
    features_to_keep = [
        'Type', # Payment type
        'Days for shipment (scheduled)', # Promised delivery window
        'Category Name',
        'Customer Segment',
        'Customer Country',
        'Market',
        'Order Region',
        'Order Country',
        'order date (DateOrders)',
        'Order Item Quantity',
        'Sales',
        'Product Price',
        'Shipping Mode',
        'Late' # Target
    ]
    
    df_clean = df[features_to_keep].copy()
    
    # 3. Handle Dates
    df_clean['order date (DateOrders)'] = pd.to_datetime(df_clean['order date (DateOrders)'])
    
    # 4. Handle Missing Values
    print("Missing values before cleaning:")
    print(df_clean.isnull().sum())
    
    # Fill or drop missing values (if any)
    df_clean = df_clean.dropna()
    
    # 5. Save cleaned data
    df_clean.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}. Shape: {df_clean.shape}")
    
    # Print data quality report
    print("\n--- Data Quality Report ---")
    print(f"Total Records: {len(df_clean)}")
    print(f"Date Range: {df_clean['order date (DateOrders)'].min()} to {df_clean['order date (DateOrders)'].max()}")
    print(f"Target Balance (Late=1): {df_clean['Late'].mean():.2%}")
    print("---------------------------\n")

if __name__ == "__main__":
    clean_data('data/raw/DataCoSupplyChainDataset.csv', 'data/processed/clean_data.csv')
