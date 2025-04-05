import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

url = "https://raw.githubusercontent.com/datasets/fifa-23/main/data/players_23.csv"
data = pd.read_csv(url)


features = ['overall', 'potential', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
target = 'team_position'  # Posición en el campo (DEL, MC, DEF, etc.)

data_clean = data.dropna(subset=features + [target])

valid_positions = ['GK', 'CB', 'LB', 'RB', 'CM', 'LM', 'RM', 'CAM', 'CDM', 'CF', 'ST', 'LW', 'RW']
data_clean = data_clean[data_clean[target].isin(valid_positions)]

X = data_clean[features]
y = data_clean[target]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("Resultados kNN:")
print(classification_report(y_test, y_pred_knn, target_names=le.classes_))
print(f"Precisión: {accuracy_score(y_test, y_pred_knn):.2f}")

svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

print("\nResultados SVM:")
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))
print(f"Precisión: {accuracy_score(y_test, y_pred_svm):.2f}")