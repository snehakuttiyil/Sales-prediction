import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
data = pd.read_csv("sales.csv")

# Features (inputs) and target (output)
X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Sales Prediction Model Trained Successfully!")
print(f"📌 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📌 R² Score: {r2:.2f}")

# Save the model
joblib.dump(model, "sales_prediction_model.pkl")
print("💾 Model saved as sales_prediction_model.pkl")

# Take user input for prediction
print("\n--- Predict Sales ---")
tv = float(input("Enter TV advertising budget: "))
radio = float(input("Enter Radio advertising budget: "))
newspaper = float(input("Enter Newspaper advertising budget: "))

prediction = model.predict([[tv, radio, newspaper]])
print(f"\n📈 Predicted Sales: {prediction[0]:.2f}")