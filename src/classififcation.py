import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

# Pfade überprüfen
print(os.getcwd())
current_path = os.getcwd()
uebergeordneter_pfad = os.path.dirname(current_path)
print("Übergeordneter Pfad:", uebergeordneter_pfad)
os.chdir('..')  # Eine Ebene zurückgehen

# Daten mit Punkt als Dezimaltrennzeichen einlesen
df = pd.read_csv('data/dataset_5secondWindow.csv', decimal='.')

# Datenvorverarbeitung
print(df.isnull().sum())

for column in df.columns:
    print(f'Spalte: {column}, Datentyp: {df[column].dtype}')

numeric_cols = df.select_dtypes(include=['float64']).columns

X = df.drop(['target', 'id', 'user'], axis=1)
y = df['target']

# Fehlende Werte durch Nullen ersetzen
X[numeric_cols] = X[numeric_cols].fillna(0)

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Cross-Validation-Setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5 Folds
accuracies = []

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"Fold {fold+1}")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Modell (KNN)
    knn = KNeighborsClassifier(n_neighbors=5)  # Du kannst n_neighbors anpassen
    knn.fit(X_train, y_train)

    # Vorhersagen auf dem Validierungsset
    y_pred = knn.predict(X_val)

    # Genauigkeit berechnen
    accuracy = accuracy_score(y_val, y_pred)
    accuracies.append(accuracy)
    print(f'Fold {fold+1} Validation Accuracy: {accuracy:.4f}')

# Durchschnittliche Genauigkeit
mean_accuracy = sum(accuracies) / len(accuracies)
print(f'Mean Cross-Validation Accuracy: {mean_accuracy:.4f}')

# Beispielvorhersagen auf dem ersten Fold-Validierungsset
knn.fit(X.iloc[train_index], y[train_index]) #Fit with the last fold train data.
predicted = knn.predict(X.iloc[val_index])
predicted_labels = label_encoder.inverse_transform(predicted)
print("Example predicted labels:", predicted_labels[:5])