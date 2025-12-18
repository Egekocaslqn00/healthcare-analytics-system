"""
Heart Disease Prediction Model
Trains and evaluates multiple ML models for heart disease prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HEART DISEASE PREDICTION MODEL")
print("="*80)

# Load data
print("\n1. Loading Data...")
df = pd.read_csv('../data/raw/heart.csv')
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts()}")

# Prepare data
print("\n2. Preparing Data...")
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, '../models/heart_scaler.pkl')
print("✓ Scaler saved")

# Define models
print("\n3. Training Models...")
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Find best model
print("\n4. Model Comparison...")
best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
best_model = results[best_model_name]['model']
print(f"\nBest Model: {best_model_name}")
print(f"ROC-AUC Score: {results[best_model_name]['roc_auc']:.4f}")

# Save best model
joblib.dump(best_model, '../models/heart_disease_model.pkl')
print(f"✓ Best model saved: heart_disease_model.pkl")

# Hyperparameter tuning for best model
print("\n5. Hyperparameter Tuning...")
if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                               param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Use tuned model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, '../models/heart_disease_model_tuned.pkl')
    print("✓ Tuned model saved")

# Detailed evaluation
print("\n6. Detailed Evaluation...")
y_pred_final = best_model.predict(X_test_scaled)
y_pred_proba_final = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=['No Disease', 'Disease']))

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
axes[0, 0].set_xlim([0.7, 1.0])
for i, v in enumerate(accuracies):
    axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center')

# ROC-AUC comparison
roc_aucs = [results[m]['roc_auc'] for m in model_names]
axes[0, 1].barh(model_names, roc_aucs, color='orange')
axes[0, 1].set_xlabel('ROC-AUC Score')
axes[0, 1].set_title('Model ROC-AUC Comparison')
axes[0, 1].set_xlim([0.7, 1.0])
for i, v in enumerate(roc_aucs):
    axes[0, 1].text(v + 0.01, i, f'{v:.3f}', va='center')

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
axes[1, 0].set_title('Confusion Matrix - Best Model')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_final)
axes[1, 1].plot(fpr, tpr, color='darkorange', lw=2, 
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
plt.savefig('../reports/figures/heart_disease_model_evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: heart_disease_model_evaluation.png")

# Feature Importance
if hasattr(best_model, 'feature_importances_'):
    fig, ax = plt.subplots(figsize=(10, 8))
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=feature_importance, y='feature', x='importance', ax=ax, palette='viridis')
    ax.set_title('Feature Importance - Heart Disease Prediction', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Features')
    plt.tight_layout()
    plt.savefig('../reports/figures/heart_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: heart_feature_importance.png")
    
    print("\nTop 5 Important Features:")
    print(feature_importance.head())

# Save results summary
results_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': [results[m]['accuracy'] for m in model_names],
    'Precision': [results[m]['precision'] for m in model_names],
    'Recall': [results[m]['recall'] for m in model_names],
    'F1-Score': [results[m]['f1'] for m in model_names],
    'ROC-AUC': [results[m]['roc_auc'] for m in model_names],
    'CV Mean': [results[m]['cv_mean'] for m in model_names],
    'CV Std': [results[m]['cv_std'] for m in model_names]
})

results_df.to_csv('../reports/heart_disease_model_results.csv', index=False)
print("\n✓ Saved: heart_disease_model_results.csv")

print("\n" + "="*80)
print("HEART DISEASE MODEL TRAINING COMPLETED!")
print("="*80)
print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba_final):.4f}")
