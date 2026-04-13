import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples (increased for better tree balancing)
n_samples = 3000

# Base generation mapping logic perfectly balanced to 3 equal classes.
# We will generate 1000 of each class and intentionally adjust features so the model can learn easily.
samples_per_class = n_samples // 3

dfs = []

for class_id in range(3):
    if class_id == 0:
        # LOW RISK: Young, low BP, low cholesterol, normal sugar, low resting HR, asymptomatic
        age = np.random.randint(20, 45, samples_per_class)
        bp = np.random.normal(110, 10, samples_per_class).astype(int)
        chol = np.random.normal(180, 20, samples_per_class).astype(int)
        bs = np.zeros(samples_per_class, dtype=int)
        hr = np.random.normal(130, 15, samples_per_class).astype(int)
        cp = np.random.choice([2, 3], size=samples_per_class) # non-anginal / asymptomatic
        angina = np.zeros(samples_per_class, dtype=int)
    
    elif class_id == 1:
        # MEDIUM RISK: Middle age, elevated BP and Chol, occasional sugar
        age = np.random.randint(45, 60, samples_per_class)
        bp = np.random.normal(135, 12, samples_per_class).astype(int)
        chol = np.random.normal(230, 25, samples_per_class).astype(int)
        bs = np.random.choice([0, 1], p=[0.7, 0.3], size=samples_per_class)
        hr = np.random.normal(150, 15, samples_per_class).astype(int)
        cp = np.random.choice([1, 2], size=samples_per_class)
        angina = np.random.choice([0, 1], p=[0.8, 0.2], size=samples_per_class)

    else:
        # HIGH RISK: Older, high BP, high Chol, likely sugar, angina
        age = np.random.randint(60, 85, samples_per_class)
        bp = np.random.normal(160, 15, samples_per_class).astype(int)
        chol = np.random.normal(280, 30, samples_per_class).astype(int)
        bs = np.random.choice([0, 1], p=[0.4, 0.6], size=samples_per_class)
        hr = np.random.normal(180, 20, samples_per_class).astype(int)
        cp = np.random.choice([0, 1], size=samples_per_class) # typical/atypical
        angina = np.random.choice([0, 1], p=[0.2, 0.8], size=samples_per_class)
        
    gender = np.random.randint(0, 2, samples_per_class)
    
    # Clip constraints to valid numeric ranges
    bp = np.clip(bp, 50, 250)
    chol = np.clip(chol, 50, 600)
    hr = np.clip(hr, 40, 220)

    risk_level = np.full(samples_per_class, class_id, dtype=int)

    # Create chunk DataFrame
    chunk = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Blood_Pressure': bp,
        'Cholesterol': chol,
        'Blood_Sugar': bs,
        'Heart_Rate': hr,
        'Chest_Pain_Type': cp,
        'Exercise_Angina': angina,
        'Risk_Level': risk_level
    })
    dfs.append(chunk)

# Concatenate and shuffle the dataset
df = pd.concat(dfs, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv('dataset.csv', index=False)
print(f"Synthetic dataset generated perfectly balanced. Shape: {df.shape}")
