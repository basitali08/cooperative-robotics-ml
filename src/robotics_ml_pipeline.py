import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("COOPERATIVE ROBOTICS ML PIPELINE")
print("Predicting Mission Success for Multi-Robot Teams")
print("=" * 80)

# ──────────── 1. LOAD DATA ────────────
print("\n" + "=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

df = pd.read_csv('data/robotics_coordination_data.csv')
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ──────────── 2. EXPLORATORY DATA ANALYSIS ────────────
print("\n" + "=" * 80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print(f"\nDataset Info:")
print(df.info())

print(f"\nDescriptive Statistics:\n{df.describe()}")

os.makedirs('results', exist_ok=True)

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('mission_success')
for idx, col in enumerate(numeric_cols):
    axes[idx].hist(df[col], bins=30, color='#3498DB', edgecolor='white', alpha=0.8)
    axes[idx].set_title(col, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('Frequency')
for idx in range(len(numeric_cols), len(axes)):
    axes[idx].axis('off')
plt.suptitle('Feature Distributions - Cooperative Robotics', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# ──────────── 3. CHECK MISSING VALUES ────────────
print("\n" + "=" * 80)
print("STEP 3: CHECKING MISSING VALUES")
print("=" * 80)

missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage': missing_pct})
print(f"\n{missing_df}")

if missing.sum() == 0:
    print("\n[OK] No missing values found in dataset")
else:
    print(f"\n[WARN] Found {missing.sum()} missing values - handling them")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    print("Missing values handled successfully")

# ──────────── 4. CHECK DUPLICATES ────────────
print("\n" + "=" * 80)
print("STEP 4: CHECKING DUPLICATES")
print("=" * 80)

duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print(f"Removed {duplicates} duplicates")

# ──────────── 5. OUTLIER DETECTION ────────────
print("\n" + "=" * 80)
print("STEP 5: OUTLIER DETECTION (IQR Method)")
print("=" * 80)

feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'mission_success']
outlier_counts = {}
for col in feature_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_counts[col] = outliers

outlier_df = pd.DataFrame(list(outlier_counts.items()), columns=['Feature', 'Outlier Count'])
print(f"\n{outlier_df}")

# ──────────── 6. ENCODE CATEGORICAL VARIABLES ────────────
print("\n" + "=" * 80)
print("STEP 6: ENCODING CATEGORICAL VARIABLES")
print("=" * 80)

df = pd.get_dummies(df, columns=['formation_type'], prefix='formation')
print(f"\nOne-hot encoded formation_type")
print(f"New columns: {[c for c in df.columns if c.startswith('formation_')]}")
print(f"Updated shape: {df.shape}")

# ──────────── 7. FEATURE ENGINEERING ────────────
print("\n" + "=" * 80)
print("STEP 7: FEATURE ENGINEERING")
print("=" * 80)

df['coord_quality_index'] = (1 - df['communication_delay'] / 500) * df['signal_strength'].apply(lambda x: 1 - (x + 120) / 90)
df['swarm_efficiency'] = df['swarm_density'] * df['energy_efficiency'] / (df['coordination_overhead'] / 40 + 0.1)
df['mission_urgency'] = df['task_complexity'] / (df['task_deadline'] / 300 + 0.1)
df['mobility_score'] = df['robot_speed'] * df['sensor_range'] / 250
df['obstacle_risk'] = df['obstacle_density'] * df['collision_probability'] / 15
df['battery_strain'] = (1 - df['battery_level'] / 100) * df['task_complexity']
df['terrain_challenge'] = df['terrain_roughness'] * (1 - df['robot_speed'] / 5)
df['robot_count_binned'] = pd.cut(df['robot_count'], bins=[1, 10, 25, 50], labels=['small', 'medium', 'large'])
df = pd.get_dummies(df, columns=['robot_count_binned'], prefix='team_size')

team_size_cols = [c for c in df.columns if c.startswith('team_size_')]
print(f"\nTeam size bins: {team_size_cols}")

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interact_base = df[['robot_count', 'task_complexity', 'communication_delay', 'obstacle_density', 'battery_level']]
interact_features = poly.fit_transform(interact_base)
interact_names = poly.get_feature_names_out(interact_base.columns)
interact_df = pd.DataFrame(interact_features, columns=interact_names, index=df.index)
interact_df = interact_df.loc[:, ~interact_df.columns.isin(interact_base.columns)]
df = pd.concat([df, interact_df.add_prefix('interact_')], axis=1)

print(f"\nInteraction features created: {len(interact_df.columns)}")
print(f"New features: coord_quality_index, swarm_efficiency, mission_urgency, mobility_score, obstacle_risk, battery_strain, terrain_challenge, team_size bins, interaction terms")
print(f"Updated shape: {df.shape}")

# ──────────── 8. CORRELATION ANALYSIS ────────────
print("\n" + "=" * 80)
print("STEP 8: CORRELATION ANALYSIS")
print("=" * 80)

numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
target_corr = corr['mission_success'].sort_values(ascending=False)
print(f"\nTop correlations with mission_success:\n{target_corr.head(10)}")
print(f"\nBottom correlations with mission_success:\n{target_corr.tail(5)}")

plt.figure(figsize=(16, 14))
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap - Cooperative Robotics', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ──────────── 9. TARGET DISTRIBUTION ────────────
print("\n" + "=" * 80)
print("STEP 9: TARGET VARIABLE ANALYSIS")
print("=" * 80)

print(f"\nMission Success Distribution:")
risk_count = df['mission_success'].value_counts()
risk_pct = df['mission_success'].value_counts(normalize=True) * 100
dist_df = pd.DataFrame({
    'Count': risk_count,
    'Percentage': risk_pct
})
print(f"\n{dist_df}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
labels = ['Failed (0)', 'Success (1)']
colors = ['#E74C3C', '#2ECC71']
axes[0].bar(labels, risk_count.values, color=colors, edgecolor='white', alpha=0.8)
axes[0].set_title('Mission Success Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(risk_count.values):
    axes[0].text(i, v + 20, f'{v}', ha='center', fontweight='bold')

axes[1].pie(risk_count.values, labels=labels, autopct='%1.1f%%', colors=colors,
            startangle=90, explode=(0.05, 0.05))
axes[1].set_title('Mission Success Proportion', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# ──────────── 10. TRAIN-TEST SPLIT ────────────
print("\n" + "=" * 80)
print("STEP 10: TRAIN-TEST SPLIT")
print("=" * 80)

X = df.drop(['mission_success'], axis=1)
y = df['mission_success']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Training set class distribution:\n{y_train.value_counts()}")
print(f"Test set class distribution:\n{y_test.value_counts()}")

# ──────────── 11. SCALING ────────────
print("\n" + "=" * 80)
print("STEP 11: FEATURE SCALING")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Features scaled using StandardScaler")
print(f"Training data shape: {X_train_scaled.shape}")
print(f"Test data shape: {X_test_scaled.shape}")

# ──────────── 12. HANDLE CLASS IMBALANCE ────────────
print("\n" + "=" * 80)
print("STEP 12: HANDLING CLASS IMBALANCE WITH SMOTE")
print("=" * 80)

print(f"Before SMOTE - Class distribution: {np.bincount(y_train)}")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE - Class distribution: {np.bincount(y_train_resampled)}")

# ──────────── 13. MODEL TRAINING ────────────
print("\n" + "=" * 80)
print("STEP 13: MODEL TRAINING")
print("=" * 80)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=12, min_samples_split=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=5, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss', use_label_encoder=False),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
}

trained_models = {}
results = []

for name, model in models.items():
    print(f"\n  Training {name}...", end=' ')

    if name == 'XGBoost':
        model.fit(X_train_resampled, y_train_resampled, verbose=False)
    else:
        model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    trained_models[name] = {'model': model, 'y_pred': y_pred, 'y_proba': y_proba}
    results.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})

    print(f"[OK] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
print(f"\n\nModel Performance Summary:\n{results_df.to_string(index=False)}")

plt.figure(figsize=(12, 6))
bar_width = 0.2
x = np.arange(len(results_df))
for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score']):
    plt.bar(x + i * bar_width, results_df[metric], bar_width, label=metric)
plt.xlabel('Models', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Comparison - Cooperative Robotics', fontsize=14, fontweight='bold')
plt.xticks(x + bar_width * 1.5, results_df['Model'], rotation=30, ha='right')
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ──────────── 14. CONFUSION MATRICES ────────────
print("\n" + "=" * 80)
print("STEP 14: CONFUSION MATRICES")
print("=" * 80)

best_model_name = results_df.iloc[0]['Model']
best_model_info = trained_models[best_model_name]

fig, axes = plt.subplots(2, 3, figsize=(16, 12))
axes = axes.flatten()
for idx, (name, info) in enumerate(trained_models.items()):
    cm = confusion_matrix(y_test, info['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlBu', ax=axes[idx],
                xticklabels=['Failed', 'Success'],
                yticklabels=['Failed', 'Success'])
    axes[idx].set_title(f'{name}\nAccuracy: {accuracy_score(y_test, info["y_pred"]):.3f}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('results/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nBest Model: {best_model_name}")
print(f"\nConfusion Matrix for {best_model_name}:")
cm_best = confusion_matrix(y_test, best_model_info['y_pred'])
cm_df = pd.DataFrame(cm_best, index=['Actual: Failed', 'Actual: Success'],
                     columns=['Predicted: Failed', 'Predicted: Success'])
print(f"\n{cm_df}")

tn, fp, fn, tp = cm_best.ravel()
print(f"\n  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives:  {tp}")

# ──────────── 15. CLASSIFICATION REPORT ────────────
print("\n" + "=" * 80)
print("STEP 15: CLASSIFICATION REPORT")
print("=" * 80)

for name, info in trained_models.items():
    print(f"\n  {name}:")
    print(f"  {classification_report(y_test, info['y_pred'], target_names=['Failed', 'Success'])}")

# ──────────── 16. ROC CURVES ────────────
print("\n" + "=" * 80)
print("STEP 16: ROC-AUC ANALYSIS")
print("=" * 80)

plt.figure(figsize=(10, 8))
roc_results = []
for name, info in trained_models.items():
    if info['y_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, info['y_proba'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        roc_results.append({'Model': name, 'ROC-AUC': roc_auc})
        print(f"  {name} - AUC: {roc_auc:.4f}")

roc_df = pd.DataFrame(roc_results).sort_values('ROC-AUC', ascending=False)

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Mission Success Prediction', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# ──────────── 17. FEATURE IMPORTANCE ────────────
print("\n" + "=" * 80)
print("STEP 17: FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

tree_models = ['Random Forest', 'Gradient Boosting', 'XGBoost']
feature_importance_data = {}

for name in tree_models:
    if name in trained_models:
        model = trained_models[name]['model']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            feature_importance_data[name] = feat_imp_df

            print(f"\n  Top 10 Features ({name}):")
            for idx, row in feat_imp_df.head(10).iterrows():
                print(f"    {row['Feature']}: {row['Importance']:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
for idx, (name, feat_df) in enumerate(feature_importance_data.items()):
    top15 = feat_df.head(15)
    axes[idx].barh(range(len(top15)), top15['Importance'].values, color='#3498DB')
    axes[idx].set_yticks(range(len(top15)))
    axes[idx].set_yticklabels(top15['Feature'].values)
    axes[idx].invert_yaxis()
    axes[idx].set_title(f'{name} - Feature Importance', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('Importance')
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ──────────── 18. CROSS VALIDATION ────────────
print("\n" + "=" * 80)
print("STEP 18: CROSS-VALIDATION (5-Fold)")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, info in trained_models.items():
    model = info['model']
    try:
        cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
        cv_results.append({
            'Model': name,
            'CV Mean': cv_scores.mean(),
            'CV Std': cv_scores.std(),
            'CV Min': cv_scores.min(),
            'CV Max': cv_scores.max()
        })
        print(f"  {name}: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f} (Min: {cv_scores.min():.4f}, Max: {cv_scores.max():.4f})")
    except:
        print(f"  {name}: Cross-validation skipped")

# ──────────── 19. SAVE MODELS ────────────
print("\n" + "=" * 80)
print("STEP 19: SAVING MODELS")
print("=" * 80)

for name, info in trained_models.items():
    safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    joblib.dump(info['model'], f'models/{safe_name}.pkl')
    print(f"  {name} saved to models/{safe_name}.pkl")

best_model = trained_models[best_model_name]['model']
os.makedirs('models', exist_ok=True)

joblib.dump(best_model, 'models/best_classifier.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(list(X.columns), 'models/feature_names.pkl')

print(f"\n  Best classifier ({best_model_name}) saved to models/best_classifier.pkl")
print(f"  Scaler saved to models/scaler.pkl")
print(f"  Feature names saved to models/feature_names.pkl")
print(f"  All individual models saved to models/")

# ──────────── 20. FINAL SUMMARY ────────────
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)

best_result = results_df.iloc[0]
print(f"\n  Best Model:                {best_result['Model']}")
print(f"  Accuracy:                  {best_result['Accuracy']:.4f} ({best_result['Accuracy']*100:.2f}%)")
print(f"  Precision:                 {best_result['Precision']:.4f}")
print(f"  Recall:                    {best_result['Recall']:.4f}")
print(f"  F1-Score:                  {best_result['F1-Score']:.4f}")

if best_model_name in feature_importance_data:
    top_feat = feature_importance_data[best_model_name].iloc[0]
    print(f"  Top Feature:               {top_feat['Feature']} ({top_feat['Importance']:.4f})")

if len(roc_results) > 0:
    best_roc = roc_df.iloc[0]
    print(f"  Best ROC-AUC:              {best_roc['Model']} ({best_roc['ROC-AUC']:.4f})")

print(f"\n  All results saved to: results/")
print(f"  All models saved to: models/")
print("=" * 80)
print("\nPipeline completed successfully!")
