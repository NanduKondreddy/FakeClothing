import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from category_encoders import TargetEncoder
from fuzzywuzzy import fuzz
import lightgbm as lgb

# Suppress LightGBM feature name warnings globally
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv(r"D:\FakeClothing\dataset\fake_clothing_scams.csv", low_memory=False)
print("✅ Dataset loaded successfully! Shape:", df.shape)

# ===============================
# 2. Feature Engineering
# ===============================
df['Price_Diff'] = df['Actual_Market_Price'] - df['Advertised_Price']
df['Price_Ratio'] = (df['Advertised_Price'] / (df['Actual_Market_Price'] + 1e-5)).clip(0, 10)

brands = ["Nike", "Adidas", "Puma", "Levi's", "Burberry", "Versace", "Gucci", "Prada"]
df['Brand_Similarity'] = df['Product_Name'].apply(
    lambda x: max([fuzz.ratio(str(x), b) for b in brands]) / 100
)

categorical_features = ['Platform', 'Product_Category', 'Payment_Method', 'Customer_Region']
numeric_features = ['Price_Diff', 'Price_Ratio', 'Customer_Age',
                    'label_logo', 'label_brand', 'label_qr', 'Brand_Similarity']

# Drop constant numeric columns
constant_cols = [col for col in numeric_features if df[col].nunique() <= 1]
if constant_cols:
    print(f"⚠ Dropping constant numeric columns: {constant_cols}")
numeric_features = [col for col in numeric_features if col not in constant_cols]

# ===============================
# 3. Train/Test Split
# ===============================
X = df[numeric_features + categorical_features]
y = df['Scam_Flag']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ===============================
# 4. Target Encoding
# ===============================
target_encoder = TargetEncoder(cols=categorical_features, smoothing=0.3)
X_train = X_train.copy()
X_test = X_test.copy()
X_train[categorical_features] = target_encoder.fit_transform(X_train[categorical_features], y_train)
X_test[categorical_features] = target_encoder.transform(X_test[categorical_features])

# ===============================
# 5. Preprocessing Pipeline
# ===============================
numeric_transformer = Pipeline([
    ('scaler', StandardScaler()),
    ('var_thresh', VarianceThreshold(threshold=1e-4))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', 'passthrough', categorical_features)
])

# ===============================
# 6. Base Models
# ===============================
bagging_lr = BaggingClassifier(
    estimator=LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
    n_estimators=10,
    n_jobs=-1,
    random_state=42
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    class_weight='balanced',
    boosting_type='gbdt',
    random_state=42,
    min_data_in_leaf=30,
    min_child_samples=30,
    force_row_wise=True,
    verbose=-1
)

# ===============================
# 7. Stacking Ensemble
# ===============================
stack_model = StackingClassifier(
    estimators=[
        ('bagging_lr', bagging_lr),
        ('lgb', lgb_model)
    ],
    final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced'),
    n_jobs=-1
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', stack_model)
])

# ===============================
# 8. Train Model
# ===============================
model.fit(X_train, y_train)
print("✅ Model training completed!")

# ===============================
# 9. Evaluation
# ===============================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(f"\n📊 Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"📈 ROC-AUC Score: {roc_auc_score(y_test, y_proba):.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# ===============================
# 10. Sample Predictions
# ===============================
print("\n🔍 Predictions on sample products:")
sample_rows = df.iloc[:5][numeric_features + categorical_features].copy()
sample_rows[categorical_features] = target_encoder.transform(sample_rows[categorical_features])

sample_preds = model.predict(sample_rows)
sample_probas = model.predict_proba(sample_rows)[:, 1]

for idx, (pred, proba) in enumerate(zip(sample_preds, sample_probas)):
    label = "Fake" if pred == 1 else "Real"
    print(f"Product: {df.iloc[idx]['Product_Name']}, Prediction: {label}, Probability: {proba:.2f}")

# ===============================
# 11. Save Model
# ===============================
joblib.dump(model, 'fake_clothing_model.pkl')
print("\n💾 Model saved as 'fake_clothing_model.pkl'")

# ===============================
# 12. Feature Importance
# ===============================
lgb_estimator = model.named_steps['classifier'].named_estimators_['lgb']
importances = lgb_estimator.feature_importances_

num_features_transformed = model.named_steps['preprocessor'].named_transformers_['num'].named_steps['var_thresh'].get_feature_names_out()
feature_names = list(num_features_transformed) + categorical_features

feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(10,6))
plt.barh(feat_imp_df['Feature'][::-1], feat_imp_df['Importance'][::-1], color='skyblue')
plt.xlabel("Importance")
plt.title("Top 15 Feature Importances (LightGBM)")
plt.tight_layout()
plt.show()