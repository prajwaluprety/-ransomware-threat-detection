import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import time

# Load data
df = pd.read_csv('ransomware_data.csv')
X = df.drop('label', axis=1)
y = df['label']

# Convert labels to numbers
le = LabelEncoder()
y = le.fit_transform(y)

# Split 80% train 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# Train the model
print("\nTraining Random Forest model...")
start = time.time()
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
elapsed = round(time.time() - start, 1)
print(f"Training complete in {elapsed} seconds!")

# Evaluate
y_pred = model.predict(X_test)
acc = round(accuracy_score(y_test, y_pred) * 100, 2)
print(f"\nAccuracy: {acc}%")
print("\nDetailed results:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix chart
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Confusion matrix — ransomware detector')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Save the model
joblib.dump(model, 'ransomware_detector.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Model saved as ransomware_detector.pkl")