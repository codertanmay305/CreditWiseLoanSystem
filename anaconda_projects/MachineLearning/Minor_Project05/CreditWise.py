"""
CreditWise Loan System - Complete ML Pipeline
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                           recall_score, f1_score, classification_report)

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 2. LOAD DATA
# ============================================================================
print("="*50)
print("LOADING DATA")
print("="*50)

df = pd.read_csv("loan_approval_data.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ============================================================================
# 3. HANDLE MISSING VALUES
# ============================================================================
print("\n" + "="*50)
print("HANDLING MISSING VALUES")
print("="*50)

print(f"Missing values before:\n{df.isnull().sum()}")

categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["float64"]).columns

# Impute numerical
num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

# Impute categorical
cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

print(f"Missing values after:\n{df.isnull().sum()}")

# ============================================================================
# 4. EXPLORATORY DATA ANALYSIS (Visualizations will be saved)
# ============================================================================
print("\n" + "="*50)
print("GENERATING VISUALIZATIONS")
print("="*50)

# Target distribution
classes_count = df["Loan_Approved"].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(classes_count, labels=["No", "Yes"], autopct="%1.1f%%", 
        startangle=90, colors=['#ff6b6b', '#4ecdc4'])
plt.title("Loan Approval Distribution", fontsize=14, fontweight='bold')
plt.savefig('loan_distribution.png')
plt.show()

print("Class Distribution:")
print(classes_count)
print(f"\nPercentage:\n{df['Loan_Approved'].value_counts(normalize=True) * 100}")

# More visualizations... (keep as is, but add plt.show() after each)

# ============================================================================
# 5. FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Drop Applicant ID
df = df.drop("Applicant_ID", axis=1)

# Label Encoding
le = LabelEncoder()
df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

# One-Hot Encoding
cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", 
        "Property_Area", "Gender", "Employer_Category"]

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(df[cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=df.index)
df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)

print(f"Shape after encoding: {df.shape}")

# ============================================================================
# 6. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*50)
print("CORRELATION ANALYSIS")
print("="*50)

num_cols = df.select_dtypes(include="number")
corr_matrix = num_cols.corr()
print("Top correlations with Loan_Approved:")
print(corr_matrix["Loan_Approved"].sort_values(ascending=False).head(10))

# ============================================================================
# 7. FIRST MODEL PIPELINE (Original features)
# ============================================================================
print("\n" + "="*50)
print("FIRST MODEL PIPELINE - ORIGINAL FEATURES")
print("="*50)

X1 = df.drop("Loan_Approved", axis=1)
y1 = df["Loan_Approved"]

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

scaler1 = StandardScaler()
X1_train_scaled = scaler1.fit_transform(X1_train)
X1_test_scaled = scaler1.transform(X1_test)

# Logistic Regression
log_model1 = LogisticRegression()
log_model1.fit(X1_train_scaled, y1_train)
y1_pred_log = log_model1.predict(X1_test_scaled)

print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy_score(y1_test, y1_pred_log):.4f}")
print(f"Precision: {precision_score(y1_test, y1_pred_log):.4f}")
print(f"Recall: {recall_score(y1_test, y1_pred_log):.4f}")
print(f"F1 Score: {f1_score(y1_test, y1_pred_log):.4f}")

# KNN
knn_model1 = KNeighborsClassifier(n_neighbors=5)
knn_model1.fit(X1_train_scaled, y1_train)
y1_pred_knn = knn_model1.predict(X1_test_scaled)

print("\nKNN Results:")
print(f"Accuracy: {accuracy_score(y1_test, y1_pred_knn):.4f}")
print(f"Precision: {precision_score(y1_test, y1_pred_knn):.4f}")
print(f"Recall: {recall_score(y1_test, y1_pred_knn):.4f}")
print(f"F1 Score: {f1_score(y1_test, y1_pred_knn):.4f}")

# Naive Bayes
nb_model1 = GaussianNB()
nb_model1.fit(X1_train_scaled, y1_train)
y1_pred_nb = nb_model1.predict(X1_test_scaled)

print("\nNaive Bayes Results:")
print(f"Accuracy: {accuracy_score(y1_test, y1_pred_nb):.4f}")
print(f"Precision: {precision_score(y1_test, y1_pred_nb):.4f}")
print(f"Recall: {recall_score(y1_test, y1_pred_nb):.4f}")
print(f"F1 Score: {f1_score(y1_test, y1_pred_nb):.4f}")

# ============================================================================
# 8. SECOND MODEL PIPELINE (With engineered features)
# ============================================================================
print("\n" + "="*50)
print("SECOND MODEL PIPELINE - WITH ENGINEERED FEATURES")
print("="*50)

# Add engineered features
df2 = df.copy()
df2["DTI_Ratio_sq"] = df2["DTI_Ratio"] ** 2
df2["Credit_Score_sq"] = df2["Credit_Score"] ** 2
df2["Applicant_Income_log"] = np.log1p(df2["Applicant_Income"])

X2 = df2.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
y2 = df2["Loan_Approved"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

# Logistic Regression
log_model2 = LogisticRegression()
log_model2.fit(X2_train_scaled, y2_train)
y2_pred_log = log_model2.predict(X2_test_scaled)

print("\nLogistic Regression Results (with engineered features):")
print(f"Accuracy: {accuracy_score(y2_test, y2_pred_log):.4f}")
print(f"Precision: {precision_score(y2_test, y2_pred_log):.4f}")

# KNN
knn_model2 = KNeighborsClassifier(n_neighbors=5)
knn_model2.fit(X2_train_scaled, y2_train)
y2_pred_knn = knn_model2.predict(X2_test_scaled)

print("\nKNN Results (with engineered features):")
print(f"Accuracy: {accuracy_score(y2_test, y2_pred_knn):.4f}")
print(f"Precision: {precision_score(y2_test, y2_pred_knn):.4f}")

# Naive Bayes
nb_model2 = GaussianNB()
nb_model2.fit(X2_train_scaled, y2_train)
y2_pred_nb = nb_model2.predict(X2_test_scaled)

print("\nNaive Bayes Results (with engineered features):")
print(f"Accuracy: {accuracy_score(y2_test, y2_pred_nb):.4f}")
print(f"Precision: {precision_score(y2_test, y2_pred_nb):.4f}")

# ============================================================================
# 9. MODEL COMPARISON
# ============================================================================
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

models = {
    "Logistic Regression": log_model2,
    "KNN": knn_model2,
    "Naive Bayes": nb_model2
}

results = {}
for name, model in models.items():
    y_pred = model.predict(X2_test_scaled)
    results[name] = {
        'Accuracy': accuracy_score(y2_test, y_pred),
        'Precision': precision_score(y2_test, y_pred),
        'Recall': recall_score(y2_test, y_pred),
        'F1': f1_score(y2_test, y_pred)
    }

results_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:")
print(results_df.round(4))

best_model = results_df['Precision'].idxmax()
print(f"\n🏆 Best Model by Precision: {best_model}")

# ============================================================================
# 10. SAVE BEST MODEL
# ============================================================================
import joblib

best_model_obj = models[best_model]
joblib.dump(best_model_obj, 'best_loan_model.pkl')
joblib.dump(scaler2, 'scaler.pkl')
print("\n✅ Best model saved as 'best_loan_model.pkl'")

print("\n" + "="*50)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*50)