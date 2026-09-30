import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.dummy import DummyClassifier

def evaluate_model(model, X, y, threshold=0.5):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    return {
        'ROC-AUC': roc_auc_score(y, y_prob),
        'PR-AUC': average_precision_score(y, y_prob),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred),
        'F1': f1_score(y, y_pred),
        'Accuracy': accuracy_score(y, y_pred)
    }

def main():
    print("Loading data...")
    df = pd.read_csv('data/processed/features_data.csv')
    df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
    
    print("Splitting data by time...")
    train_df = df[df['order date (DateOrders)'].dt.year <= 2016]
    val_df = df[df['order date (DateOrders)'].dt.year == 2017]
    test_df = df[df['order date (DateOrders)'].dt.year == 2018]
    
    # Target
    y_train = train_df['Late']
    y_val = val_df['Late']
    y_test = test_df['Late']
    
    # Drop date and target
    X_train = train_df.drop(columns=['order date (DateOrders)', 'Late'])
    X_val = val_df.drop(columns=['order date (DateOrders)', 'Late'])
    X_test = test_df.drop(columns=['order date (DateOrders)', 'Late'])
    
    # Define preprocessing
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    print("Defining models...")
    models = {
        'Majority-Class Baseline': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=50, class_weight='balanced', max_depth=10, random_state=42, n_jobs=-1))
        ]),
        'XGBoost': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=1.0, random_state=42, n_jobs=-1))
        ]) # Note: XGB handles imbalance via scale_pos_weight, but dataset is ~57% positive, so 1.0 is fine.
    }
    
    results = []
    
    print("Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        metrics = evaluate_model(model, X_val, y_val)
        metrics['Model'] = name
        results.append(metrics)
    
    # Compare
    results_df = pd.DataFrame(results)
    results_df = results_df[['Model', 'ROC-AUC', 'PR-AUC', 'Precision', 'Recall', 'F1', 'Accuracy']]
    print("\nModel Comparison (Validation Set):")
    print(results_df)
    
    results_df.to_csv('data/processed/model_comparison.csv', index=False)
    
    # Select Best Model based on PR-AUC
    best_model_name = results_df.sort_values(by='PR-AUC', ascending=False).iloc[0]['Model']
    print(f"\nSelected Best Model: {best_model_name}")
    
    best_model = models[best_model_name]
    
    # Save the best model
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
        
    print("Best model saved to models/best_model.pkl")
    
    # Threshold Analysis (Optional, saving data for evaluate.py)
    y_val_prob = best_model.predict_proba(X_val)[:, 1]
    val_results = pd.DataFrame({'true': y_val, 'prob': y_val_prob})
    val_results.to_csv('data/processed/val_predictions.csv', index=False)
    
    # Save test set for evaluate.py
    test_df.to_csv('data/processed/test_data_for_eval.csv', index=False)

if __name__ == "__main__":
    main()
