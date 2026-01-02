import zipfile
import pandas as pd
import numpy as np
import os
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


ZIP_PATH = "C:/Users/cjani/Downloads/bank+marketing.zip"


def load_all_csv_from_zip(zip_path, nested=False):
    """
    Loads ALL CSV files in the ZIP into a dictionary {filename: DataFrame}
    Handles CSV files in subdirectories and nested ZIP files recursively.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    
    dataframes = {}

    with zipfile.ZipFile(zip_path, "r") as z:
        # Get all files in the ZIP (including those in subdirectories)
        all_files = z.namelist()
        
        # Filter for CSV files (case-insensitive, handles subdirectories)
        csv_files = [f for f in all_files if f.lower().endswith(".csv") and not f.endswith("/")]
        
        # Filter for nested ZIP files
        nested_zips = [f for f in all_files if f.lower().endswith(".zip") and not f.endswith("/")]
        
        # Load CSV files directly in this ZIP
        for file in csv_files:
            with z.open(file) as f:
                df = pd.read_csv(f, sep=None, engine="python")  # auto separator
                dataframes[os.path.basename(file)] = df
        
        # Recursively load CSV files from nested ZIPs
        if nested_zips:
            if not nested:
                print(f"\nFound {len(nested_zips)} nested ZIP file(s), extracting CSV files...")
            for nested_zip_file in nested_zips:
                with z.open(nested_zip_file) as nested_zip_stream:
                    # Create a temporary file-like object from the nested ZIP
                    nested_zip_bytes = nested_zip_stream.read()
                    nested_zip_path = io.BytesIO(nested_zip_bytes)
                    
                    # Recursively load from nested ZIP
                    # zipfile.ZipFile can accept a file-like object
                    with zipfile.ZipFile(nested_zip_path, "r") as nested_z:
                        nested_all_files = nested_z.namelist()
                        nested_csv_files = [f for f in nested_all_files if f.lower().endswith(".csv") and not f.endswith("/")]
                        
                        for nested_csv_file in nested_csv_files:
                            with nested_z.open(nested_csv_file) as nested_csv_stream:
                                df = pd.read_csv(nested_csv_stream, sep=None, engine="python")
                                dataframes[os.path.basename(nested_csv_file)] = df
        
        if not csv_files and not nested_zips:
            # Provide diagnostic information
            if not nested:
                print(f"\nDebug: ZIP contains {len(all_files)} files/directories:")
                for f in all_files[:20]:  # Show first 20 entries
                    print(f"  - {f}")
                if len(all_files) > 20:
                    print(f"  ... and {len(all_files) - 20} more")
                raise ValueError(f"No CSV files or nested ZIPs found in ZIP: {zip_path}\nFound {len(all_files)} total entries.")
    
    if not nested and dataframes:
        print(f"\nFound {len(dataframes)} CSV file(s) total:")
        for filename in dataframes.keys():
            print(f"  - {filename}")

    return dataframes


# -------- LOAD DATA --------
datasets = load_all_csv_from_zip(ZIP_PATH)

# Check if any CSV files were found
if not datasets:
    raise ValueError(f"No CSV files found in ZIP: {ZIP_PATH}")

# If multiple CSVs exist and share schema, concatenate them
df_list = list(datasets.values())
if len(df_list) > 1:
    # Check if all DataFrames have the same columns
    first_cols = df_list[0].columns
    same_schema = all(
        len(d.columns) == len(first_cols) and 
        list(d.columns) == list(first_cols) 
        for d in df_list
    )
    if same_schema:
        df = pd.concat(df_list, ignore_index=True)
        print(f"\nConcatenated {len(df_list)} DataFrames with matching schemas")
    else:
        # Use the first DataFrame, or the largest one
        df = max(df_list, key=len)
        print(f"\nDataFrames have different schemas. Using the largest DataFrame ({df.shape[0]} rows)")
else:
    df = df_list[0]   # fallback


print("\n===== DATA LOADED =====")
print(df.head())


# -------- IDENTIFY TARGET COLUMN --------
possible_targets = ["y", "purchase", "bought", "will_purchase", "subscribed"]

target_col = None
for col in df.columns:
    if col.lower() in possible_targets:
        target_col = col
        break

if not target_col:
    raise ValueError("Target column not found. Expected one of: " + ", ".join(possible_targets))

print(f"\nTarget column detected: {target_col}")


# -------- HANDLE MISSING VALUES --------
for col in df.columns:
    if df[col].dtype == "O":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())


# -------- ENCODE CATEGORICAL VARIABLES --------
label_encoders = {}
for col in df.columns:
    if df[col].dtype == "O":
        le = LabelEncoder()
        
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le


# -------- SPLIT FEATURES & TARGET --------
X = df.drop(columns=[target_col])
y = df[target_col]


# -------- TRAIN / TEST SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# -------- TRAIN DECISION TREE --------
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# -------- PREDICT --------
y_pred = model.predict(X_test)


# -------- EVALUATE --------
print("\n===== MODEL PERFORMANCE =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\n===== TASK COMPLETE =====")
print("Decision Tree model successfully trained and evaluated.")
