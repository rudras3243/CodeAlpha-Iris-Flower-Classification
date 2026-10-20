# Iris Flower Classification
# Dataset: Iris.csv
# Model: Support Vector Machine (SVM)

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Load dataset
df = pd.read_csv("Iris.csv")

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nClass distribution:")
print(df["Species"].value_counts())

# 2. Select input features and target
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[features]
y = df["Species"]

# Id is intentionally excluded because it is only an identifier,
# not a biological measurement.

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Build SVM pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf"))
])

# 5. Train model
model.fit(X_train, y_train)

# 6. Predict test data
y_pred = model.predict(X_test)

# 7. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
).plot()
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.show()

# 9. Example prediction
sample = pd.DataFrame([{
    "SepalLengthCm": 5.1,
    "SepalWidthCm": 3.5,
    "PetalLengthCm": 1.4,
    "PetalWidthCm": 0.2
}])

print("\nExample flower prediction:")
print(model.predict(sample)[0])
