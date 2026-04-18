import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

print("Loading dataset...")
df = pd.read_csv('nexus_fraud_dataset.csv')

# Preprocessing
print("Preprocessing data...")
# Drop user_id as it's not a predictive feature
df = df.drop('user_id', axis=1)

# Handle categorical variable 'transaction_type'
df['transaction_type_international'] = (df['transaction_type'] == 'international').astype(int)
df = df.drop('transaction_type', axis=1)

# Separate features and target
X = df.drop('fraud_label', axis=1)
y = df['fraud_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training RandomForest model...")
# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

print("Evaluating model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("Saving model and columns...")
joblib.dump(model, 'fraud_model.joblib')
joblib.dump(list(X.columns), 'model_columns.joblib')
print("Training complete! Model saved to fraud_model.joblib")
