import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. LOAD DATASET
# ==========================================

train_df = pd.read_csv("../dataset/realistic_train_dataset.csv")
test_df = pd.read_csv("../dataset/realistic_test_dataset.csv")

features = [
    "Energy",
    "AveragePower",
    "Variance",
    "PeakAmplitude",
    "SpectralPeakRatio"
]

X_train = train_df[features]
y_train = train_df["Label"]

# ==========================================
# 2. TRAIN RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# 3. SELECT 10 CHANNELS FOR SIMULATION
# ==========================================

channels = test_df.sample(
    n=10,
    random_state=42
).copy()

X_channels = channels[features]

# ==========================================
# 4. PREDICT CHANNEL STATUS
# ==========================================

predictions = model.predict(X_channels)

# ==========================================
# 5. DISPLAY CHANNEL STATUS
# ==========================================

print("\n==========================================")
print("AI-BASED SPECTRUM SENSING")
print("==========================================")

print("\nChannel Status:")
print("------------------------------------------")

free_channels = []
busy_channels = []

for i, prediction in enumerate(predictions, start=1):

    prediction = str(prediction).upper()

    if prediction in ["FREE", "0"]:
        status = "FREE"
        free_channels.append(i)
    else:
        status = "BUSY"
        busy_channels.append(i)

    print(f"Channel {i:2d} : {status}")

# ==========================================
# 6. DYNAMIC CHANNEL ALLOCATION
# ==========================================

print("\n------------------------------------------")
print("AVAILABLE FREE CHANNELS")
print("------------------------------------------")

print(free_channels)

print("\n------------------------------------------")
print("DYNAMIC CHANNEL ALLOCATION")
print("------------------------------------------")

if free_channels:

    selected_channel = free_channels[0]

    print(f"Selected Channel : Channel {selected_channel}")
    print("Status           : ALLOCATED")
    print("Method           : Random Forest Prediction")
    print("Reason           : Channel predicted as FREE")

else:

    print("No FREE channel available.")
    print("Allocation Status : FAILED")

# ==========================================
# 7. SUMMARY
# ==========================================

print("\n==========================================")
print("SYSTEM SUMMARY")
print("==========================================")

print(f"Total Channels : {len(predictions)}")
print(f"FREE Channels  : {len(free_channels)}")
print(f"BUSY Channels  : {len(busy_channels)}")

print("\nSystem Pipeline:")
print("Signal Features")
print("      ↓")
print("Random Forest")
print("      ↓")
print("FREE / BUSY")
print("      ↓")
print("Dynamic Allocation")
print("==========================================")