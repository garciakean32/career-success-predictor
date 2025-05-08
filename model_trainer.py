# model_trainer.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("education_career_success.csv")

# Drop ID column (not useful for prediction)
df = df.drop(columns=["Student_ID"])

# Select 3 entries for input prediction (can be from anywhere)
input_df = df.sample(3, random_state=42)
df_remaining = df.drop(input_df.index)

# Save input data to file
input_df.to_csv("input_data.csv", index=False)

# Define features and target
features = [
    'Age', 'Gender', 'High_School_GPA', 'SAT_Score', 'University_Ranking',
    'University_GPA', 'Field_of_Study', 'Internships_Completed', 'Projects_Completed',
    'Certifications', 'Soft_Skills_Score', 'Networking_Score'
]
target = 'Career_Satisfaction'

# Encode categorical features
df_processed = df_remaining.copy()
for col in ['Gender', 'Field_of_Study']:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    joblib.dump(le, f"{col}_encoder.pkl")

# Train-test split
X = df_processed[features]
y = df_processed[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
    "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
    "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
}

# Save model and metrics
joblib.dump(model, "career_model.pkl")
joblib.dump(metrics, "model_metrics.pkl")

print("Model trained and saved. Evaluation metrics:")
print(metrics)