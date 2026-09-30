# Delivery Delay Prediction & Root-Cause Analytics

## 1. Project Overview
This project predicts whether a logistics order will be delivered later than the promised delivery date. It uses only information available at the time of order placement to prevent data leakage. Additionally, it identifies the key operational factors driving delivery delays using SHAP values.

## 2. Business Problem
Late deliveries negatively impact customer satisfaction and retention. By predicting late deliveries *before* they happen (at order placement), the business can proactively manage customer expectations, upgrade shipping, or investigate operational bottlenecks.

## 3. Dataset
- **Source:** DataCo Smart Supply Chain Dataset (publicly available raw CSV).
- **Records:** 180,519 rows.
- **Date Range:** 2015-01-01 to 2018-01-31.

## 4. Target Definition
**Target Variable:** `Late`
- `1` if `Days for shipping (real)` > `Days for shipment (scheduled)`
- `0` otherwise

## 5. Prediction Point
The prediction point is the moment an order is placed. The model only has access to information known at that specific time.

## 6. Leakage Prevention
Variables that represent future operational events or target-derived data were strictly excluded from the feature set:
- `Days for shipping (real)`
- `Delivery Status`
- `Late_delivery_risk`
- `shipping date (DateOrders)`
- `Order Status`

## 7. Data Quality
- No missing values in the selected feature set.
- Target Balance: 57.28% Late.

## 8. EDA & Root-Cause Analysis
- **Key Insight:** Shipping Mode heavily dictates delay risk. `Standard Class` has a ~40% late delivery rate, while `First Class`, `Second Class`, and `Same Day` have nearly 100% late delivery rates in this specific dataset (due to extremely tight scheduled days). 

## 9. Statistical Analysis
- **Research Question:** Is the late-delivery rate significantly different between Standard Class and First Class shipping?
- **Null Hypothesis:** There is no difference.
- **Alternative Hypothesis:** There is a significant difference.
- **Test:** Two-Proportion Z-Test
- **Result:** Z = -179.27, p-value = 0.0
- **Interpretation:** We reject the null hypothesis. Standard Class shipping has a significantly lower late-delivery rate than First Class shipping.

## 10. Feature Engineering
Engineered time-based features from `order date`:
- `order_month`, `order_day_of_week`, `is_weekend`, `quarter`, `peak_season`.
Kept relevant pre-order operational metrics (Sales, Order Item Quantity, Product Price, Category, Customer Location).

## 11. Model Development
Trained four models:
1. Majority-Class Baseline
2. Logistic Regression (Scaled, One-hot encoded)
3. Random Forest (Class weights balanced)
4. XGBClassifier

## 12. Model Validation
Time-aware split strategy:
- Train: 2015-2016
- Validation: 2017
- Test: 2018

## 13. Model Evaluation
Evaluated on the Validation Set:
- **Random Forest** achieved the best performance with PR-AUC of 0.8430 and ROC-AUC of 0.7430.
- Selected Threshold = `0.4` to optimize F1 (0.6965) while keeping Precision (0.83) and Recall (0.60) balanced.
- Evaluated on untouched **Test Set** (2018): Performance held steady (ROC-AUC: 0.7457).

## 14. SHAP Explainability
Global SHAP feature importance showed that the top predictors of delivery delays are:
1. Shipping Mode (Standard Class vs First Class)
2. Days for shipment (scheduled)
3. Sales amount and Order Day of Week

## 15. Business Insights
1. **Shipping Mode dictates delay risk:** Tighter delivery windows (First Class, Same Day) consistently result in late deliveries, suggesting the promised delivery windows are operationally unrealistic.
2. **Standard Class is more reliable:** Giving the carrier more scheduled days (4 days for Standard) results in a much lower late rate (~40%).
3. **Volume/Seasonality:** Order day of week and month mildly influence the risk, indicating some capacity bottlenecks.

## 16. Recommendations
1. **Recalibrate Promised Delivery Windows:** First Class and Same Day promises are currently failing. The business should dynamically pad promised delivery dates based on origin/destination pairs rather than hard-coding them by Shipping Mode.
2. **Prioritize High-Risk Orders:** If the Prediction Studio flags a high-value order as "High Risk", customer service can proactively notify the customer, or operations can expedite the fulfillment process.

## 17. Web Application (Prediction Studio)
A Flask + HTML/JS web application was built to demonstrate the model in action. It allows a user to input pre-order information and receives:
1. Probability of Delay
2. Risk Classification
3. Local SHAP explanation detailing *why* this specific order is high/low risk.

## 18. Project Architecture
```
data/
    raw/ (Raw CSV)
    processed/ (Cleaned data, test sets, predictions)
src/
    data_cleaning.py
    feature_engineering.py
    statistics.py
    train.py
    evaluate.py
    explainability.py
models/
    best_model.pkl
frontend/
    index.html
    styles.css
    app.js
backend/
    app.py
```

## 19. How to Run
1. Install requirements: `pip install pandas numpy scikit-learn xgboost shap flask matplotlib statsmodels`
2. Run backend: `python backend/app.py`
3. Navigate to `http://localhost:5001`

## 20. Limitations
- The underlying DataCo dataset has highly deterministic rules for generating `Days for shipping (real)` which makes the prediction heavily reliant on Shipping Mode. 
- Geographic distance wasn't explicitly modeled due to lack of reliable lat/lon pairs for both origin and destination.

## 21. Future Improvements
- Integrate real-time weather and traffic data at the time of order placement.
- Use a dataset with more organic variability in delivery times.

## 22. Interview Explanation (2-3 Minutes)
"I built a delivery delay prediction and root-cause analytics system to predict whether an order would be delivered later than promised. I started by analyzing historical order and delivery data to understand where delays were concentrated. I defined late delivery as actual delivery occurring after the promised delivery date. Since the model is intended to make a prediction before delivery happens, I ensured no post-delivery variables were used. I engineered time and operational features available at order placement and performed a two-proportion Z-test, which proved a statistically significant difference in delay rates between shipping modes.

For modeling, I compared Logistic Regression, Random Forest, and XGBoost using a time-aware split, selecting Random Forest based on PR-AUC. I manually tuned the probability threshold to balance precision and recall. I then used SHAP to understand why individual orders received higher risk scores. Finally, I built a web application where users can enter order information, receive a late-delivery probability, and see the specific operational factors contributing to that prediction."
