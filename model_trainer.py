import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("education_career_success.csv")
df = df.drop(columns=["Student_ID"])

# Extract 3 entries for prediction only
input_df = df.sample(3, random_state=42)
df_remaining = df.drop(input_df.index)
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
gender_encoder = LabelEncoder()
field_encoder = LabelEncoder()
df_processed['Gender'] = gender_encoder.fit_transform(df_processed['Gender'])
df_processed['Field_of_Study'] = field_encoder.fit_transform(df_processed['Field_of_Study'])

# Save encoders
joblib.dump(gender_encoder, "Gender_encoder.pkl")
joblib.dump(field_encoder, "Field_of_Study_encoder.pkl")

# Train-test split
X = df_processed[features]
y = df_processed[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

metrics_all = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    metrics_all[name] = metrics
    joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.pkl")

# ✅ Save Decision Tree as the default model
joblib.dump(models["Decision Tree"], "career_model.pkl")
joblib.dump(metrics_all["Decision Tree"], "model_metrics.pkl")
joblib.dump(metrics_all, "all_model_metrics.pkl")
