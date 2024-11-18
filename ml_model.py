import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load your data (example: logs_df.csv)
log_df = pd.read_csv('logs_df(ml).csv')

# Feature engineering (assemble features)
feature_cols = ['Method', 'Endpoint', 'Protocol', 'Content Size', 'No of Requests']
X = log_df[feature_cols]
y = log_df['Status Code']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
}

# Train each classifier, evaluate accuracy, save the models, and print results
for name, clf in classifiers.items():
    # Train the model
    clf.fit(X_train_scaled, y_train)
    
    # Save the trained model
    model_path = f"/home/saliniyan/Documents/BDA/{name.replace(' ', '_')}_model.pkl"
    joblib.dump(clf, model_path)
    print(f"{name} model saved at {model_path}")
    
    # Save the scaler
    scaler_path = f"/home/saliniyan/Documents/BDA/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved at {scaler_path}")
    
    # Make predictions
    y_pred = clf.predict(X_test_scaled)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
