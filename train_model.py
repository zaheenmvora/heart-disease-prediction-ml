import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/heart_disease_data.csv")

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

# Manual Prediction (Insert your own data)
# Example patient data
# Format must match dataset column order
input_data = (52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3)

# Convert to numpy array
input_data_array = np.asarray(input_data)

# Reshape for single prediction
input_data_reshaped = input_data_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

print("\nPrediction Result:")

if prediction[0] == 0:
    print("The person does NOT have Heart Disease")
else:
    print("The person HAS Heart Disease")
