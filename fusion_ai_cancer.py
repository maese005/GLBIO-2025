# Step 1: Generate Dummy Multimodal Data
##############################
# This will simulate clinical, imaging, and genomic data for multiple patients.

# Import modules
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import shap

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples (patients)
num_samples = 1000

# Generate Clinical Data
clinical_data = pd.DataFrame({
    "Patient_ID": range(num_samples),
    "Age": np.random.randint(30, 85, size=num_samples),
    "ECOG_Score": np.random.choice([0, 1, 2, 3, 4], size=num_samples),
    "Comorbidities": np.random.choice([0, 1], size=num_samples),  # 0 = No, 1 = Yes
})

# Generate Imaging Data (CT-based body composition, scaled to arbitrary units)
imaging_data = pd.DataFrame({
    "Patient_ID": range(num_samples),
    "Muscle_Mass": np.random.uniform(10, 50, size=num_samples),  
    "Visceral_Fat": np.random.uniform(5, 30, size=num_samples),  
    "Subcutaneous_Fat": np.random.uniform(10, 40, size=num_samples),
})

# Generate Genomic Data (Mutations as binary indicators)
genomic_data = pd.DataFrame({
    "Patient_ID": range(num_samples),
    "TP53_Mut": np.random.choice([0, 1], size=num_samples),  # 1 = mutation present
    "EGFR_Mut": np.random.choice([0, 1], size=num_samples),
    "KRAS_Mut": np.random.choice([0, 1], size=num_samples),
})

# Merge all datasets on Patient_ID
merged_df = clinical_data.merge(imaging_data, on="Patient_ID").merge(genomic_data, on="Patient_ID")

# Generate Survival Outcome (Binary classification: 1 = Survived, 0 = Died)
merged_df["Survival"] = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])  # 70% survival rate

# Display a sample of the final dataset
print(merged_df.head())

# Step 2: Preprocessing
##############################
# Now, we normalize numerical features and one-hot encode categorical ones.

# Select feature columns
numerical_features = ["Age", "Muscle_Mass", "Visceral_Fat", "Subcutaneous_Fat"]
categorical_features = ["ECOG_Score", "Comorbidities", "TP53_Mut", "EGFR_Mut", "KRAS_Mut"]
target_column = "Survival"

# Scale numerical features
scaler = StandardScaler()
merged_df[numerical_features] = scaler.fit_transform(merged_df[numerical_features])

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False)
encoded_cat_features = encoder.fit_transform(merged_df[categorical_features])
encoded_cat_df = pd.DataFrame(encoded_cat_features, columns=encoder.get_feature_names_out(categorical_features))

# Concatenate processed data
X = pd.concat([merged_df[numerical_features], encoded_cat_df], axis=1)
y = merged_df[target_column]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Step 3: Build a Multimodal Deep Learning Model (Early Fusion)
##############################
# We concatenate all feature types and pass them through a neural network.

# Define input shape
input_shape = X_train.shape[1]

# Build the deep learning model
model = keras.Sequential([
    layers.Input(shape=(input_shape,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # Binary classification (Survival)
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 4: Model Explainability using SHAP
# To make the model interpretable, we use SHAP (SHapley Additive Explanations).

# Create SHAP explainer
explainer = shap.Explainer(model, X_train)

# Compute SHAP values
shap_values = explainer(X_test)

# Visualize feature importance
shap.summary_plot(shap_values, X_test)

