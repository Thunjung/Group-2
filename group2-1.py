import pandas as pd

# Load the dataset
df = pd.read_csv("house_price_missing_letters_lasso_friendly.csv")

# Columns to clean
cols_to_check = ['Area', 'Bedrooms', 'Age']

# Function to check if a value is numeric
def is_numeric(val):
    try:
        float(val)
        return True
    except:
        return False

# Keep rows where all three columns are numeric
mask = df[cols_to_check].applymap(is_numeric).all(axis=1)
filtered_df = df[mask].copy()  # .copy() to avoid SettingWithCopyWarning

# Convert numeric columns to float
filtered_df[cols_to_check] = filtered_df[cols_to_check].astype(float)

# Save cleaned data to CSV
filtered_df.to_csv("house_price_clean_numeric.csv", index=False)

# Print the cleaned data
print("Cleaned data:")
print(filtered_df)  # This will print the entire cleaned dataset
