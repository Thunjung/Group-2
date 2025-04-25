import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Load the dataset
try:
    df = pd.read_csv("house_price_clean_numeric.csv")
except FileNotFoundError:
    print("Error: File 'house_price_clean_numeric.csv' not found.")
    exit()

# 2. Convert columns with mixed data to string type
for col in ['Area', 'Bedrooms', 'Age']:
    df[col] = df[col].astype(str)

# 3. Separate features and target
X = df.drop(columns=["Price"])
y = df["Price"]

# 4. One-hot encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# 5. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# 6. Train the Lasso regression model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# 7. Predict and calculate RMSE
y_pred = lasso.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 8. Create a DataFrame of feature importances
coef_df = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Coefficient": lasso.coef_
})
coef_df["Importance"] = coef_df["Coefficient"].abs()
ranked_features = coef_df[coef_df["Coefficient"] != 0].sort_values(by="Importance", ascending=False)

# 9. Display results
print("Intercept:", round(lasso.intercept_, 2))
print("RMSE on test set:", round(rmse, 2))

# Configure pandas to display the entire DataFrame without truncation
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

print("\nRanked features by importance:")
print(ranked_features)

# 10. (Optional) Save the results to a CSV file
ranked_features.to_csv("lasso_feature_importance.csv", index=False)
print("\nFeature importances have been saved to 'lasso_feature_importance.csv'")
