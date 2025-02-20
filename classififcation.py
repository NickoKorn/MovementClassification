import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Daten mit Punkt als Dezimaltrennzeichen einlesen
df = pd.read_csv('dataset_5secondWindow.csv', decimal='.')

# 2. Preprocess the Data

print(df.isnull().sum())

for column in df.columns:
    
    print(f'Spalte: {column}, Datentyp: {df[column].dtype}')

numeric_cols = df.select_dtypes(include=['float64']).columns

X = df.drop(['target', 'id', 'user'], axis=1)
y = df['target']
X.dropna()

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols]) #enabled scaling

# Korrelationsmatrix berechnen
correlation_matrix = X[numeric_cols].corr()

# Korrelationsmatrix anzeigen
print(correlation_matrix)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)

class SimpleClassifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_size = X_train.shape[1]
num_classes = len(label_encoder.classes_)
model = SimpleClassifier(input_size, num_classes)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0000001)

num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f'Test Accuracy: {accuracy:.4f}')

predicted_labels = label_encoder.inverse_transform(predicted.numpy())
print("Example predicted labels:", predicted_labels[:5])