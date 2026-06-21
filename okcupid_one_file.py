# OKCupid Date A Scientist Project
# One-file notebook-style script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

sns.set(style="whitegrid")

# 1) Load data
# Save your dataset as profiles.csv in the same folder as this script.
df = pd.read_csv("profiles.csv")
print("Shape:", df.shape)
print(df.head())

# 2) Inspect missing values
print("
Missing values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# 3) Set target and features
target = "sex"
possible_features = [
    "age", "orientation", "body_type", "diet", "drinks", "drugs",
    "education", "ethnicity", "job", "location", "offspring",
    "pets", "religion", "sign", "smokes", "status", "income"
]
features = [col for col in possible_features if col in df.columns]
data = df[features + [target]].dropna(subset=[target]).copy()
print("
Data used for modeling:", data.shape)

# 4) EDA
if target in data.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(data=data, x=target)
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig("target_distribution.png", dpi=150)
    plt.close()

if "age" in data.columns:
    plt.figure(figsize=(7,4))
    sns.histplot(data=data, x="age", bins=30, kde=True)
    plt.title("Age Distribution")
    plt.tight_layout()
    plt.savefig("age_distribution.png", dpi=150)
    plt.close()

# 5) Train/test split
X = data.drop(columns=[target])
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6) Baseline
baseline_accuracy = y_train.value_counts(normalize=True).max()
print("Baseline accuracy:", baseline_accuracy)

# 7) Preprocessing
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 8) Logistic regression model
log_reg_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

log_reg_model.fit(X_train, y_train)
y_pred = log_reg_model.predict(X_test)

log_reg_acc = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", log_reg_acc)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 9) Random forest model
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_acc)
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# 10) Compare models
results = pd.DataFrame({
    "model": ["baseline", "logistic regression", "random forest"],
    "accuracy": [baseline_accuracy, log_reg_acc, rf_acc]
})
print(results)
results.to_csv("model_results.csv", index=False)

plt.figure(figsize=(7,4))
sns.barplot(data=results, x="model", y="accuracy")
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

# 11) Feature importance for logistic regression
try:
    ohe = log_reg_model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_names)
    coef = log_reg_model.named_steps["classifier"].coef_[0]

    feature_importance = pd.DataFrame({
        "feature": all_feature_names,
        "coefficient": coef
    })
    feature_importance["abs_coef"] = feature_importance["coefficient"].abs()
    feature_importance = feature_importance.sort_values("abs_coef", ascending=False)
    feature_importance.to_csv("feature_importance.csv", index=False)

    top_features = feature_importance.head(15).sort_values("coefficient")
    plt.figure(figsize=(10,6))
    sns.barplot(data=top_features, x="coefficient", y="feature")
    plt.title("Top Logistic Regression Features")
    plt.tight_layout()
    plt.savefig("top_features.png", dpi=150)
    plt.close()
except Exception as e:
    print("Feature importance step skipped:", e)

print("
Done. Files saved: target_distribution.png, age_distribution.png, model_comparison.png, model_results.csv, feature_importance.csv, top_features.png")
