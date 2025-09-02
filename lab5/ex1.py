import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

titanic = sns.load_dataset('titanic')

#print(titanic.head())

missing_values = titanic.isnull().sum()
missing_columns = missing_values[missing_values > 0]

df = titanic

# 1. Completează valorile lipsă pentru coloana 'age' cu media
df['age'] = df['age'].fillna(df['age'].mean())

# 2. Completează valorile lipsă pentru coloana 'embarked' cu valoarea cea mai frecventă (mode)
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# 3. Aplică One-Hot Encoding pe coloana 'embarked'
df_encoded = pd.get_dummies(df, columns=['embarked'], drop_first=True)

# 4. Aplică Label Encoding pe coloana 'sex'
encoder = LabelEncoder()
df_encoded['sex'] = encoder.fit_transform(df_encoded['sex'])

# 5. Normalizarea variabilelor numerice 'age' și 'fare'
scaler = MinMaxScaler()
df_encoded[['age', 'fare']] = scaler.fit_transform(df_encoded[['age', 'fare']])

# 6. Standardizarea variabilelor numerice 'age' și 'fare'
scaler = StandardScaler()
df_encoded[['age', 'fare']] = scaler.fit_transform(df_encoded[['age', 'fare']])

# Vizualizează primele 5 rânduri după preprocesare
#print(df_encoded.head())

# Împărțirea datelor în seturi de antrenament (80%) și seturi de test (20%)
X = df_encoded.drop(columns=['survived', 'class', 'who', 'deck', 'embark_town', 'alive'], axis=1)  # Caracteristicile (fără coloana 'survived')
y = df_encoded['survived']  # Eticheta (variabila țintă)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verifică dimensiunile seturilor
"""print("Setul de antrenament X:", X_train.shape)
print("Setul de test X:", X_test.shape)
print("Setul de antrenament y:", y_train.shape)
print("Setul de test y:", y_test.shape)"""


# Creează și antrenează modelul de regresie logistică
model = LogisticRegression(max_iter=1000)

# Antrenăm modelul
# TODO 6

# Prezicem etichetele pe setul de test
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Calculăm și afișăm acuratețea modelului
accuracy = accuracy_score(y_test, y_pred)
"""print(f"Acuratețea modelului de regresie logistică: {accuracy:.2f}")

# Afișăm coeficienții modelului pentru fiecare caracteristică (comentați aceste linii dacă folosiți alt tip de clasificator)
print("\nCoeficienții modelului de regresie logistică:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")"""
    

cm = confusion_matrix(y_test, y_pred)

# Vizualizăm matricea de confuzie
"""plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title("Matricea de confuzie")
plt.xlabel('Predicții')
plt.ylabel('Realitate')
plt.show()"""


"""# Alegem cele două caracteristici relevante: 'age' și 'fare'
plt.figure(figsize=(8, 6))

# Scatter plot cu culori diferite pentru supraviețuitori (1) și non-supraviețuitori (0)
sns.scatterplot(data=df, x='age', y='fare', hue='survived', palette='coolwarm', style='survived', markers={0: 'o', 1: 's'}, s=100)

# Setăm titlul și etichetele axelor
plt.title("Scatter Plot între Age și Fare (Supraviețuitori vs Non-Supraviețuitori)")
plt.xlabel('Age')
plt.ylabel('Fare')

# Afișăm legenda
plt.legend(title="Supraviețuitor", loc='best')

# Afișăm graficul
plt.show()"""


"""# Creăm un pairplot cu 'age' și 'fare', colorat pe baza variabilei 'survived'
sns.pairplot(df[['age', 'fare', 'survived']], hue='survived', palette='coolwarm')

# Setăm titlul graficului
plt.suptitle("Pair Plot între Age și Fare (Supraviețuitori vs Non-Supraviețuitori)", y=1.02)

# Afișăm graficul
plt.show()"""


# Creează și antrenează modelul SVM
svm_model = SVC(kernel='linear')  # Alegem kernel liniar
svm_model.fit(X_train, y_train)  # Antrenăm modelul

# Prezicem etichetele pe setul de test
y_pred_svm = svm_model.predict(X_test)

# Calculăm și afișăm acuratețea modelului SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Acuratețea modelului SVM: {accuracy_svm:.2f}")

"""# Afișăm matricea de confuzie pentru modelul SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=["Nu a supraviețuit", "A supraviețuit"])
disp_svm.plot(cmap=plt.cm.Blues)
plt.title("Matricea de confuzie - SVM")
plt.show()"""

# Creează și antrenează modelul Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Setăm 100 de arbori
rf_model.fit(X_train, y_train)  # Antrenăm modelul

# Prezicem etichetele pe setul de test
y_pred_rf = rf_model.predict(X_test)

# Calculăm și afișăm acuratețea modelului Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Acuratețea modelului Random Forest: {accuracy_rf:.2f}")

# Afișăm matricea de confuzie pentru modelul Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["Nu a supraviețuit", "A supraviețuit"])
disp_rf.plot(cmap=plt.cm.Blues)
plt.title("Matricea de confuzie - Random Forest")
plt.show()