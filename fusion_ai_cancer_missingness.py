# -------- Imports -----------------------------------------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from packaging import version
import sklearn
import shap
import matplotlib.pyplot as plt   # needed for SHAP plotting

np.random.seed(42)

# -------- Step 1: Generate synthetic multimodal data ------------------------
N = 1_000

clinical = pd.DataFrame({
    "Patient_ID": range(N),
    "Age": np.random.randint(30, 85, size=N),
    "ECOG_Score": np.random.choice([0, 1, 2, 3, 4], size=N),
    "Comorbidities": np.random.choice([0, 1], size=N),
})

imaging = pd.DataFrame({
    "Patient_ID": range(N),
    "Muscle_Mass":      np.random.uniform(10, 50, size=N),
    "Visceral_Fat":     np.random.uniform(5, 30,  size=N),
    "Subcutaneous_Fat": np.random.uniform(10, 40, size=N),
})

genomic = pd.DataFrame({
    "Patient_ID": range(N),
    "TP53_Mut": np.random.choice([0, 1], size=N),
    "EGFR_Mut": np.random.choice([0, 1], size=N),
    "KRAS_Mut": np.random.choice([0, 1], size=N),
})

df = (
    clinical
    .merge(imaging, on="Patient_ID")
    .merge(genomic,  on="Patient_ID")
)

# Binary outcome (70 % survival)
df["Survival"] = np.random.choice([0, 1], size=N, p=[0.3, 0.7])

# -------- Step 2: Inject missingness ----------------------------------------
num_cols = ["Age", "Muscle_Mass", "Visceral_Fat", "Subcutaneous_Fat"]
cat_cols = ["ECOG_Score", "Comorbidities", "TP53_Mut", "EGFR_Mut", "KRAS_Mut"]

for col in num_cols:
    df.loc[df.sample(frac=0.20, random_state=1).index, col] = np.nan

for col in cat_cols:
    df.loc[df.sample(frac=0.10, random_state=2).index, col] = np.nan

# -------- Step 3: Pre‑processing pipeline -----------------------------------
if version.parse(sklearn.__version__) >= version.parse("1.2"):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
else:
    ohe = OneHotEncoder(sparse=False,        handle_unknown="ignore")

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale",  StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", add_indicator=True)),
        ("onehot", ohe),
    ]), cat_cols),
])

X = preprocess.fit_transform(df)
y = df["Survival"]

# Capture feature names for SHAP
feature_names = preprocess.get_feature_names_out()

# Convert to DataFrames (labels preserved)
X_df = pd.DataFrame(X, columns=feature_names)

# -------- Step 4: Train/test split -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.20, random_state=42
)

# -------- Step 5: Build & train NN -----------------------------------------
input_shape = X_train.shape[1]

model = keras.Sequential([
    layers.Input(shape=(input_shape,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1,  activation="sigmoid"),
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=2
)

# -------- Step 6: Explain with SHAP ----------------------------------------
explainer   = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# SHAP summary plot  (comment if running head‑less)
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.tight_layout()
plt.show()
