"""
Healthcare Analytics System - Data Exploration
This notebook explores all four healthcare datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("HEALTHCARE ANALYTICS SYSTEM - DATA EXPLORATION")
print("="*80)

# Load all datasets
print("\n1. LOADING DATASETS...")
print("-"*80)

heart_df = pd.read_csv('../data/raw/heart.csv')
diabetes_df = pd.read_csv('../data/raw/diabetes.csv')
stroke_df = pd.read_csv('../data/raw/stroke.csv')

print(f"✓ Heart Disease Dataset: {heart_df.shape[0]} rows, {heart_df.shape[1]} columns")
print(f"✓ Diabetes Dataset: {diabetes_df.shape[0]} rows, {diabetes_df.shape[1]} columns")
print(f"✓ Stroke Dataset: {stroke_df.shape[0]} rows, {stroke_df.shape[1]} columns")

# HEART DISEASE DATASET ANALYSIS
print("\n" + "="*80)
print("2. HEART DISEASE DATASET ANALYSIS")
print("="*80)

print("\nDataset Info:")
print(heart_df.info())

print("\nBasic Statistics:")
print(heart_df.describe())

print("\nMissing Values:")
print(heart_df.isnull().sum())

print("\nTarget Distribution:")
print(heart_df['target'].value_counts())
print(f"Disease Rate: {heart_df['target'].mean()*100:.2f}%")

# DIABETES DATASET ANALYSIS
print("\n" + "="*80)
print("3. DIABETES DATASET ANALYSIS")
print("="*80)

print("\nDataset Info:")
print(diabetes_df.info())

print("\nBasic Statistics:")
print(diabetes_df.describe())

print("\nMissing Values:")
print(diabetes_df.isnull().sum())

print("\nTarget Distribution:")
print(diabetes_df['Outcome'].value_counts())
print(f"Diabetes Rate: {diabetes_df['Outcome'].mean()*100:.2f}%")

# STROKE DATASET ANALYSIS
print("\n" + "="*80)
print("4. STROKE DATASET ANALYSIS")
print("="*80)

print("\nDataset Info:")
print(stroke_df.info())

print("\nBasic Statistics:")
print(stroke_df.describe())

print("\nMissing Values:")
print(stroke_df.isnull().sum())

print("\nTarget Distribution:")
print(stroke_df['stroke'].value_counts())
print(f"Stroke Rate: {stroke_df['stroke'].mean()*100:.2f}%")

# Create visualizations
print("\n" + "="*80)
print("5. CREATING VISUALIZATIONS...")
print("="*80)

# Heart Disease Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Heart Disease Dataset - Feature Distributions', fontsize=16, fontweight='bold')

# Age distribution
axes[0, 0].hist(heart_df['age'], bins=20, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

# Sex distribution
heart_df['sex'].value_counts().plot(kind='bar', ax=axes[0, 1], color=['lightcoral', 'lightblue'])
axes[0, 1].set_title('Gender Distribution')
axes[0, 1].set_xlabel('Sex (0=Female, 1=Male)')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_xticklabels(['Female', 'Male'], rotation=0)

# Target distribution
heart_df['target'].value_counts().plot(kind='bar', ax=axes[0, 2], color=['salmon', 'lightgreen'])
axes[0, 2].set_title('Heart Disease Distribution')
axes[0, 2].set_xlabel('Target (0=No Disease, 1=Disease)')
axes[0, 2].set_ylabel('Count')
axes[0, 2].set_xticklabels(['No Disease', 'Disease'], rotation=0)

# Cholesterol distribution
axes[1, 0].hist(heart_df['chol'], bins=30, color='orange', edgecolor='black')
axes[1, 0].set_title('Cholesterol Distribution')
axes[1, 0].set_xlabel('Cholesterol (mg/dl)')
axes[1, 0].set_ylabel('Frequency')

# Max heart rate distribution
axes[1, 1].hist(heart_df['thalach'], bins=30, color='purple', edgecolor='black')
axes[1, 1].set_title('Max Heart Rate Distribution')
axes[1, 1].set_xlabel('Max Heart Rate')
axes[1, 1].set_ylabel('Frequency')

# Chest pain type
heart_df['cp'].value_counts().plot(kind='bar', ax=axes[1, 2], color='teal')
axes[1, 2].set_title('Chest Pain Type Distribution')
axes[1, 2].set_xlabel('Chest Pain Type')
axes[1, 2].set_ylabel('Count')

plt.tight_layout()
plt.savefig('../reports/figures/heart_disease_exploration.png', dpi=300, bbox_inches='tight')
print("✓ Saved: heart_disease_exploration.png")

# Diabetes Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Diabetes Dataset - Feature Distributions', fontsize=16, fontweight='bold')

# Glucose distribution
axes[0, 0].hist(diabetes_df['Glucose'], bins=30, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Glucose Distribution')
axes[0, 0].set_xlabel('Glucose Level')
axes[0, 0].set_ylabel('Frequency')

# BMI distribution
axes[0, 1].hist(diabetes_df['BMI'], bins=30, color='green', edgecolor='black')
axes[0, 1].set_title('BMI Distribution')
axes[0, 1].set_xlabel('BMI')
axes[0, 1].set_ylabel('Frequency')

# Outcome distribution
diabetes_df['Outcome'].value_counts().plot(kind='bar', ax=axes[0, 2], color=['salmon', 'lightgreen'])
axes[0, 2].set_title('Diabetes Outcome Distribution')
axes[0, 2].set_xlabel('Outcome (0=No Diabetes, 1=Diabetes)')
axes[0, 2].set_ylabel('Count')
axes[0, 2].set_xticklabels(['No Diabetes', 'Diabetes'], rotation=0)

# Age distribution
axes[1, 0].hist(diabetes_df['Age'], bins=20, color='orange', edgecolor='black')
axes[1, 0].set_title('Age Distribution')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Frequency')

# Blood Pressure distribution
axes[1, 1].hist(diabetes_df['BloodPressure'], bins=30, color='purple', edgecolor='black')
axes[1, 1].set_title('Blood Pressure Distribution')
axes[1, 1].set_xlabel('Blood Pressure')
axes[1, 1].set_ylabel('Frequency')

# Insulin distribution
axes[1, 2].hist(diabetes_df['Insulin'], bins=30, color='red', edgecolor='black')
axes[1, 2].set_title('Insulin Distribution')
axes[1, 2].set_xlabel('Insulin Level')
axes[1, 2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('../reports/figures/diabetes_exploration.png', dpi=300, bbox_inches='tight')
print("✓ Saved: diabetes_exploration.png")

# Stroke Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Stroke Dataset - Feature Distributions', fontsize=16, fontweight='bold')

# Age distribution
axes[0, 0].hist(stroke_df['age'], bins=30, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

# Gender distribution
stroke_df['gender'].value_counts().plot(kind='bar', ax=axes[0, 1], color=['lightcoral', 'lightblue', 'lightgreen'])
axes[0, 1].set_title('Gender Distribution')
axes[0, 1].set_xlabel('Gender')
axes[0, 1].set_ylabel('Count')

# Stroke distribution
stroke_df['stroke'].value_counts().plot(kind='bar', ax=axes[0, 2], color=['salmon', 'lightgreen'])
axes[0, 2].set_title('Stroke Distribution')
axes[0, 2].set_xlabel('Stroke (0=No, 1=Yes)')
axes[0, 2].set_ylabel('Count')
axes[0, 2].set_xticklabels(['No Stroke', 'Stroke'], rotation=0)

# BMI distribution
axes[1, 0].hist(stroke_df['bmi'].dropna(), bins=30, color='green', edgecolor='black')
axes[1, 0].set_title('BMI Distribution')
axes[1, 0].set_xlabel('BMI')
axes[1, 0].set_ylabel('Frequency')

# Avg glucose level distribution
axes[1, 1].hist(stroke_df['avg_glucose_level'], bins=30, color='orange', edgecolor='black')
axes[1, 1].set_title('Average Glucose Level Distribution')
axes[1, 1].set_xlabel('Avg Glucose Level')
axes[1, 1].set_ylabel('Frequency')

# Smoking status
stroke_df['smoking_status'].value_counts().plot(kind='bar', ax=axes[1, 2], color='purple')
axes[1, 2].set_title('Smoking Status Distribution')
axes[1, 2].set_xlabel('Smoking Status')
axes[1, 2].set_ylabel('Count')
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('../reports/figures/stroke_exploration.png', dpi=300, bbox_inches='tight')
print("✓ Saved: stroke_exploration.png")

# Correlation Analysis
print("\n" + "="*80)
print("6. CORRELATION ANALYSIS")
print("="*80)

# Heart Disease Correlation
fig, ax = plt.subplots(figsize=(12, 10))
correlation_matrix = heart_df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Heart Disease Dataset - Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../reports/figures/heart_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: heart_correlation.png")

# Diabetes Correlation
fig, ax = plt.subplots(figsize=(10, 8))
correlation_matrix = diabetes_df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Diabetes Dataset - Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../reports/figures/diabetes_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: diabetes_correlation.png")

# Summary Statistics
print("\n" + "="*80)
print("7. SUMMARY STATISTICS")
print("="*80)

summary_data = {
    'Dataset': ['Heart Disease', 'Diabetes', 'Stroke'],
    'Total Samples': [len(heart_df), len(diabetes_df), len(stroke_df)],
    'Features': [heart_df.shape[1]-1, diabetes_df.shape[1]-1, stroke_df.shape[1]-1],
    'Positive Cases': [heart_df['target'].sum(), diabetes_df['Outcome'].sum(), stroke_df['stroke'].sum()],
    'Positive Rate (%)': [
        f"{heart_df['target'].mean()*100:.2f}",
        f"{diabetes_df['Outcome'].mean()*100:.2f}",
        f"{stroke_df['stroke'].mean()*100:.2f}"
    ],
    'Missing Values': [
        heart_df.isnull().sum().sum(),
        diabetes_df.isnull().sum().sum(),
        stroke_df.isnull().sum().sum()
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n", summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('../reports/dataset_summary.csv', index=False)
print("\n✓ Saved: dataset_summary.csv")

print("\n" + "="*80)
print("DATA EXPLORATION COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nGenerated Files:")
print("  - heart_disease_exploration.png")
print("  - diabetes_exploration.png")
print("  - stroke_exploration.png")
print("  - heart_correlation.png")
print("  - diabetes_correlation.png")
print("  - dataset_summary.csv")
