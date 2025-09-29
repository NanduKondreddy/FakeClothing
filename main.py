import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from category_encoders import TargetEncoder
from fuzzywuzzy import fuzz
import lightgbm as lgb

# ===============================
# 1. Load Dataset
# ===============================
dataset_path = r"D:\FakeClothing\dataset\fake_clothing_scams.csv"
df = pd.read_csv(dataset_path)
print("✅ Dataset loaded successfully! Shape:", df.shape)

# ===============================
# 2. Feature Engineering
# ===============================
# Price-based features
df['Price_Diff'] = df['Actual_Market_Price'] - df['Advertised_Price']
df['Price_Ratio'] = df['Advertised_Price'] / (df['Actual_Market_Price'] + 1e-5)

# Fuzzy brand similarity
brands = ["Nike", "Adidas", "Puma", "Levi's", "Burberry", "Versace", "Gucci", "Prada"]
df['Brand_Similarity'] = df['Product_Name'].apply(lambda x: max([fuzz.ratio(x, b) for b in brands]))

# Target encode categorical features
categorical_features = ['Platform', 'Product_Category', 'Payment_Method', 'Customer_Region']
target_encoder = TargetEncoder(cols=categorical_features)
df[categorical_features] = target_encoder.fit_transform(df[categorical_features], df['Scam_Flag'])

# Interaction and numeric features
numeric_features = ['Price_Diff', 'Price_Ratio', 'Customer_Age',
                    'label_logo', 'label_brand', 'label_qr', 'Brand_Similarity']

# Final features and target
X = df[numeric_features + categorical_features]
y = df['Scam_Flag']

# ===============================
# 3. Preprocessing Pipeline
# ===============================
numeric_transformer = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', 'passthrough', categorical_features)  # already target encoded
])

# ===============================
# 4. Base Models
# ===============================
# Bagging Logistic Regression
bagging_lr = BaggingClassifier(
    estimator=LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
    n_estimators=15,
    n_jobs=-1,
    random_state=42
)

# LightGBM Classifier
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    class_weight='balanced',
    random_state=42
)

# ===============================
# 5. Stacking Ensemble
# ===============================
stack_model = StackingClassifier(
    estimators=[
        ('bagging_lr', bagging_lr),
        ('lgb', lgb_model)
    ],
    final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced'),
    n_jobs=-1
)

# Full pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', stack_model)
])

# ===============================
# 6. Train/Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ===============================
# 7. Train Model
# ===============================
model.fit(X_train, y_train)

# ===============================
# 8. Evaluation
# ===============================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n📊 Stacked Bagging LR + LightGBM Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# ===============================
# 9. Sample Predictions
# ===============================
print("\n🔍 Predictions on sample products:")
sample_rows = X.head(5)
sample_preds = model.predict(sample_rows)
for idx, pred in enumerate(sample_preds):
    label = "Fake" if pred == 1 else "Real"
    print(f"Product: {df.iloc[idx]['Product_Name']}, Prediction: {label}")
