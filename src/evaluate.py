import pandas as pd
import numpy as np
import json
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve

def main():
    print("Loading validation predictions for threshold analysis...")
    val_preds = pd.read_csv('data/processed/val_predictions.csv')
    y_val = val_preds['true']
    y_prob = val_preds['prob']
    
    thresholds = np.arange(0.1, 0.9, 0.1)
    best_threshold = 0.5
    best_f1 = 0
    
    print("\nThreshold Analysis (Validation Set):")
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 45)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_val, y_pred, zero_division=0)
        r = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        print(f"{t:<10.1f} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")
        
        if f1 > best_f1 and p > 0.6:
            best_f1 = f1
            best_threshold = t
            
    print(f"\nSelected Threshold (max F1): {best_threshold:.1f}")
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv('data/processed/test_data_for_eval.csv')
    X_test = test_df.drop(columns=['Late'])
    y_test = test_df['Late']
    
    # Load model
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    print("Evaluating on Test Set...")
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)
    
    results = {
        'ROC-AUC': roc_auc_score(y_test, test_prob),
        'PR-AUC': average_precision_score(y_test, test_prob),
        'Precision': precision_score(y_test, test_pred, zero_division=0),
        'Recall': recall_score(y_test, test_pred),
        'F1': f1_score(y_test, test_pred),
        'Accuracy': accuracy_score(y_test, test_pred),
        'Brier Score': brier_score_loss(y_test, test_prob)
    }
    
    print("\nFinal Test Set Performance:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
        
    with open('data/processed/test_results.json', 'w') as f:
        json.dump({'threshold': float(best_threshold), 'metrics': results}, f, indent=4)
        
    # Calibration Curve data
    prob_true, prob_pred = calibration_curve(y_test, test_prob, n_bins=10)
    calib_df = pd.DataFrame({'prob_true': prob_true, 'prob_pred': prob_pred})
    calib_df.to_csv('data/processed/calibration_curve.csv', index=False)
    print("Saved test results and calibration curve.")

if __name__ == "__main__":
    main()
