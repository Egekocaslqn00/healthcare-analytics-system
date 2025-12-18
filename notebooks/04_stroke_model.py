"""
Stroke Prediction Model
Trains and evaluates multiple ML models for stroke prediction
Handles imbalanced dataset using SMOTE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
# from imblearn.over_sampling import SMOTE  # Using manual oversampling instead
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STROKE PREDICTION MODEL")
print("="*80)

# Load data
print("\n1. Loading Data...")
df = pd.read_csv('../data/raw/stroke.csv')
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['stroke'].value_counts()}")
print(f"Stroke rate: {df['stroke'].mean()*100:.2f}% (HIGHLY IMBALANCED)")

# Data preprocessing
print("\n2. Data Preprocessing...")

# Drop id column
df = df.drop('id', axis=1)

# Handle missing values in BMI
df['bmi'].fillna(df['bmi'].median(), inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("✓ Categorical variables encoded")
print("✓ Missing values handled")

# Prepare data
X = df.drop('stroke', axis=1)
y = df['stroke']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and encoders
joblib.dump(scaler, '../models/stroke_scaler.pkl')
joblib.dump(label_encoders, '../models/stroke_encoders.pkl')
print("✓ Scaler and encoders saved")

# Apply manual oversampling to handle imbalanced data
print("\n3. Applying Oversampling for Imbalanced Data...")
# Find minority class
minority_class = y_train.value_counts().idxmin()
majority_class = y_train.value_counts().idxmax()

# Get boolean masks
minority_mask = y_train == minority_class
majority_mask = y_train == majority_class

# Get data
X_minority = X_train_scaled[minority_mask]
X_majority = X_train_scaled[majority_mask]
y_minority = y_train[minority_mask]
y_majority = y_train[majority_mask]

# Oversample minority class
np.random.seed(42)
oversample_size = len(y_majority)
oversample_indices = np.random.choice(len(X_minority), size=oversample_size, replace=True)
X_minority_oversampled = X_minority[oversample_indices]
y_minority_oversampled = y_minority.iloc[oversample_indices]

# Combine
X_train_resampled = np.vstack([X_majority, X_minority_oversampled])
y_train_resampled = pd.concat([y_majority, y_minority_oversampled])

print(f"Original training set: {X_train_scaled.shape}")
print(f"Resampled training set: {X_train_resampled.shape}")
print(f"Class distribution after oversampling:")
print(y_train_resampled.value_counts())

# Define models
print("\n4. Training Models...")
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model on resampled data
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predictions on original test set
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

# Find best model
print("\n5. Model Comparison...")
best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
best_model = results[best_model_name]['model']
print(f"\nBest Model: {best_model_name}")
print(f"ROC-AUC Score: {results[best_model_name]['roc_auc']:.4f}")

# Save best model
joblib.dump(best_model, '../models/stroke_model.pkl')
print(f"✓ Best model saved: stroke_model.pkl")

# Detailed evaluation
print("\n6. Detailed Evaluation...")
y_pred_final = best_model.predict(X_test_scaled)
y_pred_proba_final = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=['No Stroke', 'Stroke'], zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
print("\nConfusion Matrix:")
print(cm)

# Visualizations
print("\n7. Creating Visualizations...")

# Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Accuracy comparison
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
axes[0, 0].barh(model_names, accuracies, color='steelblue')
axes[0, 0].set_xlabel('Accuracy')
axes[0, 0].set_title('Model Accuracy Comparison')
axes[0, 0].set_xlim([0.5, 1.0])
for i, v in enumerate(accuracies):
    axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

# ROC-AUC comparison
roc_aucs = [results[m]['roc_auc'] for m in model_names]
axes[0, 1].barh(model_names, roc_aucs, color='orange')
axes[0, 1].set_xlabel('ROC-AUC Score')
axes[0, 1].set_title('Model ROC-AUC Comparison')
axes[0, 1].set_xlim([0.5, 1.0])
for i, v in enumerate(roc_aucs):
    axes[0, 1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0],
            xticklabels=['No Stroke', 'Stroke'],
            yticklabels=['No Stroke', 'Stroke'])
axes[1, 0].set_title('Confusion Matrix - Best Model')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_final)
axes[1, 1].plot(fpr, tpr, color='red', lw=2, 
                label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_proba_final):.3f})')
axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
axes[1, 1].set_xlim([0.0, 1.0])
axes[1, 1].set_ylim([0.0, 1.05])
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curve - Best Model')
axes[1, 1].legend(loc="lower right")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../reports/figures/stroke_model_evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: stroke_model_evaluation.png")

# Feature Importance
if hasattr(best_model, 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(10, 8))
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=feature_importance, y='feature', x='importance', ax=ax, palette='viridis')
    ax.set_title('Feature Importance - Stroke Prediction', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Features')
    plt.tight_layout()
    plt.savefig('../reports/figures/stroke_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: stroke_feature_importance.png")
    
    print("\nTop 5 Important Features:")
    print(feature_importance.head())

# Save results summary
results_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': [results[m]['accuracy'] for m in model_names],
    'Precision': [results[m]['precision'] for m in model_names],
    'Recall': [results[m]['recall'] for m in model_names],
    'F1-Score': [results[m]['f1'] for m in model_names],
    'ROC-AUC': [results[m]['roc_auc'] for m in model_names]
})

results_df.to_csv('../reports/stroke_model_results.csv', index=False)
print("\n✓ Saved: stroke_model_results.csv")

print("\n" + "="*80)
print("STROKE MODEL TRAINING COMPLETED!")
print("="*80)
print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba_final):.4f}")
print("\nNote: Manual oversampling was used to handle class imbalance in training data")
