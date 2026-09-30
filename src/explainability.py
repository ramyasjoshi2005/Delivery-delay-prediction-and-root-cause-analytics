import pandas as pd
import numpy as np
import pickle
import shap
import json
import matplotlib.pyplot as plt

def main():
    print("Loading test data and model for explainability...")
    test_df = pd.read_csv('data/processed/test_data_for_eval.csv')
    X_test = test_df.drop(columns=['Late'])
    
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    
    print("Transforming test data...")
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names
    cat_features = preprocessor.transformers_[1][1].get_feature_names_out()
    num_features = preprocessor.transformers_[0][2]
    all_features = list(num_features) + list(cat_features)
    
    # Sample a subset to compute SHAP values quickly (e.g., 500 rows)
    np.random.seed(42)
    sample_indices = np.random.choice(X_test_transformed.shape[0], size=500, replace=False)
    X_test_sample = X_test_transformed[sample_indices]
    
    print("Computing SHAP values...")
    # TreeExplainer is fast for tree-based models
    if type(classifier).__name__ in ['RandomForestClassifier', 'XGBClassifier']:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_test_sample)
    else:
        # Fallback for Logistic Regression
        explainer = shap.LinearExplainer(classifier, X_test_transformed)
        shap_values = explainer.shap_values(X_test_sample)
    
    # Handle if shap_values has a 3rd dimension for classes (e.g., RF in newer shap versions)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
    
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, feature_names=all_features, show=False)
    plt.tight_layout()
    plt.savefig('data/processed/shap_summary.png')
    plt.close()
    
    # Calculate global feature importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0).flatten()
    importance_df = pd.DataFrame({'feature': all_features, 'importance': mean_abs_shap})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    
    top_features = importance_df.head(10).to_dict(orient='records')
    
    with open('data/processed/shap_importance.json', 'w') as f:
        json.dump(top_features, f, indent=4)
        
    print("Saved SHAP summary plot and top features.")

if __name__ == "__main__":
    main()
