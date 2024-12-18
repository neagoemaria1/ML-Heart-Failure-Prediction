
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
import xgboost as xgb
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Preprocesarea datelor
df["diabetes"] = df["diabetes"].replace(0, "non-diabetes").replace(1, "diabetes")
df["sex"] = df["sex"].replace(0, "male").replace(1, "female")
df["smoking"] = df["smoking"].replace(0, "non-smoking").replace(1, "smoking")
df["DEATH_EVENT"] = df["DEATH_EVENT"].replace(0, False).replace(1, True)

# Observam distributia pacientilor pe anumite caracteristici
fig = px.sunburst(
    df,
    path=['sex', 'smoking', 'diabetes', 'DEATH_EVENT'],
    title="Distributia pacientilor pe baza caracteristicilor",
)
fig.show()

# Selectam caracteristicile si tinta
features = df.drop("DEATH_EVENT", axis=1)
target = df["DEATH_EVENT"]

# Codificarea variabilelor categorice
features = pd.get_dummies(features, drop_first=True)

# Impartim datele in seturi de antrenament si test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, stratify=target, random_state=42)

# Aplicam SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardizare
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Model Decision Tree
dt_model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

dt_model.fit(X_train_resampled, y_train_resampled)
y_pred_dt = dt_model.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
print("\nDecision Tree Results:")
print(f"Accuracy: {accuracy_dt:.2f}")
print(classification_report(y_test, y_pred_dt))

# Plot pentru matricea de confuzie Decision Tree
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', xticklabels=['Alive', 'Death'], yticklabels=['Alive', 'Death'])
plt.title('Confusion Matrix Decision Tree')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Plot pentru arborele de decizie
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=features.columns, class_names=['Alive', 'Death'], filled=True, rounded=True)
plt.title("Decision Tree")
plt.show()

# Model XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.2,
    min_child_weight=2,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Antrenam modelul XGBoost
xgb_model.fit(X_train_scaled, y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test_scaled)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
print("\nXGBoost Results:")
print(f"Accuracy: {accuracy_xgb:.2f}")
print(classification_report(y_test, y_pred_xgb))

# Plot pentru matricea de confuzie XGBoost
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=['Alive', 'Death'], yticklabels=['Alive', 'Death'])
plt.title('Confusion Matrix XGBoost')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

plt.subplots_adjust(top=0.85)

# Observam distributia pacientilor în functie de trombocite si deces
plt.figure(figsize=(10, 6))
plt.hist(df.loc[df['DEATH_EVENT'] == True, 'platelets'],
         bins=30, alpha=0.7, label='Decedat', color='red')
plt.hist(df.loc[df['DEATH_EVENT'] == False, 'platelets'],
         bins=30, alpha=0.7, label='Supravietuitor', color='green')
plt.xlabel('Numar de trombocite')
plt.ylabel('Numarul de pacienti')
plt.title('Distributia nr de trombocite a pacienților in funcție de evenimentul de deces')
plt.legend(title='Eveniment de deces', loc='upper right')
plt.grid(True)
plt.show()

plt.subplots_adjust(top=0.85)

# Observam distributia pacientilor in functie de varsta si deces
plt.figure(figsize=(10, 6))
plt.hist([df.loc[df.DEATH_EVENT == e, 'age'] for e in [True, False]],
         bins=30, alpha=0.7, label=['Decedat', 'Supravietuitor'])
plt.xlabel('Varsta')
plt.ylabel('Numarul de pacienți')
plt.title('Distributia varstei pacientilor in funcție de evenimentul de deces')
plt.legend(title='Eveniment de deces', loc='upper right')
plt.grid(True)
plt.show()