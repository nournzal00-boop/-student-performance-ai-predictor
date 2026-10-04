import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Dataset Generation (Simulated Academic Data)
data = {
    'math_score': [85, 45, 92, 60, 78, 30, 95, 50, 88, 40],
    'physics_score': [80, 50, 90, 55, 82, 35, 98, 48, 85, 42],
    'study_hours_per_week': [15, 5, 20, 8, 12, 3, 22, 6, 16, 4],
    'status': ['Pass', 'Fail', 'Pass', 'Pass', 'Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Fail']
}

df = pd.DataFrame(data)

# Convert target label to binary (1: Pass, 0: Fail)
df['target'] = df['status'].map({'Pass': 1, 'Fail': 0})

# 2. Features and Target Definition
X = df[['math_score', 'physics_score', 'study_hours_per_week']]
y = df['target']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 6. Predict for a New Student
new_student = [[82, 75, 14]]  # Math: 82, Physics: 75, Study Hours: 14
prediction = model.predict(new_student)
result = "Pass" if prediction[0] == 1 else "Fail"
print(f"\nPrediction for New Student: {result}")
