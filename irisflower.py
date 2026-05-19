import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
iris = pd.read_csv("iris project data.csv")

# Display first 5 rows
print("First 5 Rows:\n")
print(iris.head())

# Remove Id column
iris = iris.drop("Id", axis=1)

# Check missing values
print("\nMissing Values:\n")
print(iris.isnull().sum())

# Dataset summary
print("\nDataset Summary:\n")
print(iris.describe())

# Data Visualization
sns.pairplot(iris, hue="Species")
plt.show()

# Convert Species names into numbers
encoder = LabelEncoder()
iris["Species"] = encoder.fit_transform(iris["Species"])

# Split features and target
X = iris.drop("Species", axis=1)
y = iris["Species"]

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy * 100)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

# Plot Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Custom Prediction
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

species = encoder.inverse_transform(prediction)

print("\nPredicted Species:")
print(species[0])