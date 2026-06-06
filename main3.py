# Credit Scoring Model
# Predicting creditworthiness using classification algorithms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, 
                             confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# 1. CREATE SYNTHETIC DATASET
# ============================================

def create_credit_dataset(n_samples=10000):
    """
    Create a synthetic credit scoring dataset
    Features:
    - income: Annual income in thousands
    - age: Age in years
    - debts: Total debts in thousands
    - payment_history: 0=bad, 1=average, 2=good, 3=excellent
    - credit_utilization: Credit used / credit limit (0-1)
    - late_payments: Number of late payments in last 2 years
    - employed_years: Years at current job
    - num_credit_lines: Number of active credit lines
    """
    
    np.random.seed(42)
    
    # Generate features
    income = np.random.normal(60, 30, n_samples)
    age = np.random.randint(18, 70, n_samples)
    debts = np.random.exponential(20, n_samples)
    payment_history = np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.3, 0.2])
    credit_utilization = np.random.beta(2, 5, n_samples)
    late_payments = np.random.poisson(1.5, n_samples)
    employed_years = np.random.exponential(5, n_samples)
    num_credit_lines = np.random.poisson(3, n_samples)
    
    # Calculate credit score (target variable)
    # Higher score = better creditworthiness
    credit_score = (
        (income / 100) * 2 +
        (age / 70) * 10 +
        - (debts / 50) * 15 +
        payment_history * 25 +
        (1 - credit_utilization) * 20 +
        - late_payments * 10 +
        (employed_years / 20) * 10 +
        (num_credit_lines / 10) * 5 +
        np.random.normal(0, 5, n_samples)
    )
    
    # Binary classification: 1 = Good credit, 0 = Bad credit
    threshold = np.percentile(credit_score, 35)  # 35% bad, 65% good
    credit_worthy = (credit_score > threshold).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'income': np.round(income, 2),
        'age': age,
        'debts': np.round(debts, 2),
        'payment_history': payment_history,
        'credit_utilization': np.round(credit_utilization, 3),
        'late_payments': late_payments,
        'employed_years': np.round(employed_years, 1),
        'num_credit_lines': num_credit_lines,
        'credit_worthy': credit_worthy
    })
    
    return df

# Create the dataset
print("Creating synthetic credit dataset...")
df = create_credit_dataset(10000)
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:\n{df['credit_worthy'].value_counts()}")
print(f"Good credit (1): {df['credit_worthy'].sum()/len(df)*100:.1f}%")
print(f"Bad credit (0): {(1-df['credit_worthy'].mean())*100:.1f}%")

# Display first few rows
print("\nFirst 5 rows of dataset:")
print(df.head())

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# ============================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Correlation analysis
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=100)
plt.show()

# Feature distributions by creditworthiness
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
features = ['income', 'age', 'debts', 'payment_history', 
            'credit_utilization', 'late_payments', 'employed_years', 'num_credit_lines']

for idx, feature in enumerate(features):
    row = idx // 4
    col = idx % 4
    for credit_class in [0, 1]:
        subset = df[df['credit_worthy'] == credit_class][feature]
        axes[row, col].hist(subset, alpha=0.6, label=f'Credit: {credit_class}', bins=30)
    axes[row, col].set_title(feature)
    axes[row, col].legend()
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=100)
plt.show()

# ============================================
# 3. DATA PREPROCESSING
# ============================================

print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

# Separate features and target
X = df.drop('credit_worthy', axis=1)
y = df['credit_worthy']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures scaled using StandardScaler")

# ============================================
# 4. MODEL TRAINING AND EVALUATION
# ============================================

print("\n" + "="*50)
print("MODEL TRAINING AND EVALUATION")
print("="*50)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
}

# Store results
results = {}
predictions = {}
probabilities = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n--- {name} ---")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    predictions[name] = y_pred
    probabilities[name] = y_pred_proba
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"CV ROC-AUC (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================
# 5. MODEL COMPARISON VISUALIZATION
# ============================================

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

# Create comparison DataFrame
comparison_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:")
print(comparison_df.round(4))

# Bar plot comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

for idx, metric in enumerate(metrics):
    row = idx // 3
    col = idx % 3
    bars = axes[row, col].bar(comparison_df.index, comparison_df[metric], 
                              color=['#3498db', '#2ecc71', '#e74c3c'])
    axes[row, col].set_title(f'{metric} Comparison')
    axes[row, col].set_ylim([0, 1])
    axes[row, col].set_ylabel(metric)
    axes[row, col].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')

# Remove empty subplot
axes[1, 2].remove()
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=100)
plt.show()

# ROC Curves
plt.figure(figsize=(10, 8))
for name in models.keys():
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    roc_auc = results[name]['ROC-AUC']
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('roc_curves.png', dpi=100)
plt.show()

# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (name, model) in enumerate(models.items()):
    cm = confusion_matrix(y_test, predictions[name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Bad Credit', 'Good Credit'],
                yticklabels=['Bad Credit', 'Good Credit'])
    axes[idx].set_title(f'{name}\nConfusion Matrix')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=100)
plt.show()

# ============================================
# 6. FEATURE IMPORTANCE (Random Forest)
# ============================================

print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Get feature importance from Random Forest
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], 
         color='#3498db')
plt.xlabel('Importance', fontsize=12)
plt.title('Random Forest Feature Importance', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100)
plt.show()

# ============================================
# 7. HYPERPARAMETER TUNING (Optional Enhancement)
# ============================================

print("\n" + "="*50)
print("HYPERPARAMETER TUNING - Random Forest")
print("="*50)

# Simple grid search for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

# Use a smaller grid for faster execution
simple_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15],
    'min_samples_split': [2, 5]
}

print("Performing Grid Search (this may take a moment)...")
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                          simple_param_grid, 
                          cv=3, 
                          scoring='roc_auc',
                          n_jobs=-1,
                          verbose=1)
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

# Evaluate best model
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test_scaled)
y_pred_proba_best = best_rf.predict_proba(X_test_scaled)[:, 1]

print("\nTuned Random Forest Performance on Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_best):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_best):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_best):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_best):.4f}")

# ============================================
# 8. SAVE MODEL FOR PRODUCTION
# ============================================

print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)

import joblib

# Save the best model and scaler
joblib.dump(best_rf, 'credit_scoring_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model saved as 'credit_scoring_model.pkl'")
print("Scaler saved as 'scaler.pkl'")

# ============================================
# 9. PREDICTION FUNCTION FOR NEW DATA
# ============================================

def predict_creditworthiness(income, age, debts, payment_history, 
                            credit_utilization, late_payments, 
                            employed_years, num_credit_lines):
    """
    Predict creditworthiness for a new individual
    
    Parameters:
    - income: Annual income in thousands
    - age: Age in years
    - debts: Total debts in thousands
    - payment_history: 0=bad, 1=average, 2=good, 3=excellent
    - credit_utilization: Credit used / credit limit (0-1)
    - late_payments: Number of late payments in last 2 years
    - employed_years: Years at current job
    - num_credit_lines: Number of active credit lines
    
    Returns:
    - prediction: 0 = Bad Credit, 1 = Good Credit
    - probability: Probability of being creditworthy
    """
    
    # Load model and scaler
    model = joblib.load('credit_scoring_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Create feature array
    features = np.array([[income, age, debts, payment_history, 
                         credit_utilization, late_payments, 
                         employed_years, num_credit_lines]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    return prediction, probability

# Example prediction
print("\n" + "="*50)
print("EXAMPLE PREDICTION")
print("="*50)

# Example 1: Good candidate
print("\nExample 1 - Good Candidate:")
income, age, debts = 85, 35, 15
payment_history, credit_util = 3, 0.25
late_payments, employed_years = 0, 5
num_credit_lines = 4

pred, prob = predict_creditworthiness(income, age, debts, payment_history,
                                      credit_util, late_payments, 
                                      employed_years, num_credit_lines)
print(f"Income: ${income}k, Age: {age}, Debts: ${debts}k")
print(f"Payment History: {payment_history}, Credit Util: {credit_util:.0%}")
print(f"Prediction: {'Good Credit' if pred == 1 else 'Bad Credit'}")
print(f"Confidence: {prob:.2%}")

# Example 2: Poor candidate
print("\nExample 2 - Poor Candidate:")
income, age, debts = 30, 22, 45
payment_history, credit_util = 0, 0.95
late_payments, employed_years = 8, 0.5
num_credit_lines = 1

pred, prob = predict_creditworthiness(income, age, debts, payment_history,
                                      credit_util, late_payments, 
                                      employed_years, num_credit_lines)
print(f"Income: ${income}k, Age: {age}, Debts: ${debts}k")
print(f"Payment History: {payment_history}, Credit Util: {credit_util:.0%}")
print(f"Prediction: {'Good Credit' if pred == 1 else 'Bad Credit'}")
print(f"Confidence: {prob:.2%}")

print("\n" + "="*50)
print("CREDIT SCORING MODEL COMPLETED SUCCESSFULLY!")
print("="*50)