
# Importare librarii
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.express as px

# Importare functii
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# importare model
from sklearn.linear_model import LogisticRegression

#Incarcam datasetul si il citim
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Preprocesarea datelor
df["diabetes"] = df["diabetes"].replace(0, "non-diabetes").replace(1, "diabetes")
df["sex"] = df["sex"].replace(0, "male").replace(1, "female")
df["smoking"] = df["smoking"].replace(0, "non-smoking").replace(1, "smoking")
df["DEATH_EVENT"] = df["DEATH_EVENT"].replace(0, False).replace(1, True)

# Observam distributia pacientilor pe anumite features
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

# Impartim datelor în seturi de antrenament si test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=20)

# Crearea modelului de regresie logistica
model = LogisticRegression()

# Antrenam modelului
model.fit(X_train, y_train)

# Predictie pe setul de test
y_pred = model.predict(X_test)

# Evaluarea modelului
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Afisarea rezultatelor
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Observam distributia pacientilor in functie de tromobocite si deces
plt.figure(figsize=(10, 6))
plt.hist(df.loc[df['DEATH_EVENT'] == True, 'platelets'],
         bins=30, alpha=0.7, label='Decedat', color='red')
plt.hist(df.loc[df['DEATH_EVENT'] == False, 'platelets'],
         bins=30, alpha=0.7, label='Supravietuitor', color='green')
plt.xlabel('Numar de trombocite')
plt.ylabel('Numarul de pacienti')
plt.title('Distributia nr de trombocite a pacientilor în functie de evenimentul de deces')
plt.legend(title='Eveniment de deces', loc='upper right')
plt.grid(True)
plt.show()

plt.subplots_adjust(top=0.85)

# Observam distributia pacientilor in functie de varsta si deces
plt.figure(figsize=(10, 6))
plt.hist([df.loc[df.DEATH_EVENT == e, 'age'] for e in [True, False]],
         bins=30, alpha=0.7, label=['Decedat', 'Supravietuitor'])
plt.xlabel('Varsta')
plt.ylabel('Numarul de pacienti')
plt.title('Distributia varstei pacientilor în functie de evenimentul de deces')
plt.legend(title='Eveniment de deces', loc='upper right')
plt.grid(True)
plt.show()

plt.subplots_adjust(top=0.85)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='RdYlBu',
            xticklabels=['Death', 'Alive'],
            yticklabels=['Death', 'Alive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()