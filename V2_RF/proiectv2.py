
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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
features = df[['age', 'platelets', 'diabetes', 'sex', 'smoking']]
target = df['DEATH_EVENT']

# Codificarea variabilelor categorice
features = pd.get_dummies(features, drop_first=True)

# Impartim datele in seturi de antrenament si test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=20)

# Creem modelul Random Forest cu hiperparametrii specifici
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)

# Antrenam modelul
rf.fit(X_train, y_train)

# Predictia pe setul de test
y_pred = rf.predict(X_test)

# Evaluam modelului
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Afisam rezultatele
print(f'Acuratețe Random Forest Classifier: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Observam distributia pacientilor în functie de trombocite si deces
plt.figure(figsize=(10, 6))
plt.hist(df.loc[df['DEATH_EVENT'] == True, 'platelets'],
         bins=30, alpha=0.7, label='Decedat', color='red')
plt.hist(df.loc[df['DEATH_EVENT'] == False, 'platelets'],
         bins=30, alpha=0.7, label='Supraviețuitor', color='green')
plt.xlabel('Numar de trombocite')
plt.ylabel('Numarul de pacienti')
plt.title('Distribuția nr de trombocite a pacienților în funcție de evenimentul de deces')
plt.legend(title='Eveniment de deces', loc='upper right')
plt.grid(True)
plt.show()

plt.subplots_adjust(top=0.85)

# Observam distributia pacientilor in functie de varsta si deces
plt.figure(figsize=(10, 6))
plt.hist([df.loc[df.DEATH_EVENT == e, 'age'] for e in [True, False]],
         bins=30, alpha=0.7, label=['Decedat', 'Supraviețuitor'])
plt.xlabel('Varsta')
plt.ylabel('Numarul de pacienți')
plt.title('Distributia varstei pacientilor în funcție de evenimentul de deces')
plt.legend(title='Eveniment de deces', loc='upper right')
plt.grid(True)
plt.show()

plt.subplots_adjust(top=0.85)

# Heatmap pentru matricea de confuzie
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='RdYlBu',
            xticklabels=['Death', 'Alive'],
            yticklabels=['Death', 'Alive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()