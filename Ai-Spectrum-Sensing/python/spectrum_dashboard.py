import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. LOAD DATA
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
# 3. SELECT CHANNELS
# ==========================================

channels = test_df.sample(
    n=10,
    random_state=42
).copy()

predictions = model.predict(channels[features])

# ==========================================
# 4. CONVERT PREDICTIONS
# ==========================================

status = []

for prediction in predictions:

    prediction = str(prediction).upper()

    if prediction in ["FREE", "0"]:
        status.append("FREE")
    else:
        status.append("BUSY")

# ==========================================
# 5. FIND FREE CHANNELS
# ==========================================

free_channels = [
    i + 1 for i, s in enumerate(status)
    if s == "FREE"
]

# Allocate first free channel
if free_channels:
    allocated_channel = free_channels[0]
else:
    allocated_channel = None

# ==========================================
# 6. DISPLAY RESULT
# ==========================================

print("\n==========================================")
print("FINAL SPECTRUM SENSING DASHBOARD")
print("==========================================")

for i, s in enumerate(status, start=1):

    if i == allocated_channel:
        print(f"Channel {i:2d} : {s} → ALLOCATED")
    else:
        print(f"Channel {i:2d} : {s}")

print("------------------------------------------")

if allocated_channel:
    print(f"Allocated Channel : Channel {allocated_channel}")
else:
    print("No channel available for allocation.")

# ==========================================
# 7. CREATE VISUALIZATION
# ==========================================

values = []

for s in status:

    if s == "FREE":
        values.append(0)
    else:
        values.append(1)

plt.figure(figsize=(12, 6))

bars = plt.bar(
    range(1, 11),
    values
)

# Highlight allocated channel
if allocated_channel:
    bars[allocated_channel - 1].set_edgecolor("black")
    bars[allocated_channel - 1].set_linewidth(4)

plt.xticks(range(1, 11))
plt.yticks([0, 1], ["FREE", "BUSY"])

plt.xlabel("Channel Number")
plt.ylabel("Channel Status")
plt.title("AI-Based Cognitive Radio Spectrum Dashboard")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "../dataset/final_spectrum_dashboard.png",
    dpi=300
)

plt.show()

print("\nDashboard saved as:")
print("final_spectrum_dashboard.png")