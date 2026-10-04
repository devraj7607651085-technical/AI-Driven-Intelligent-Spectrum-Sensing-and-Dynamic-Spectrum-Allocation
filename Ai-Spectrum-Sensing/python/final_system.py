import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

X_test = test_df[features]
y_test = test_df["Label"]

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
# 3. OVERALL TEST ACCURACY
# ==========================================

test_prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, test_prediction)

# ==========================================
# 4. SELECT 10 CHANNELS
# ==========================================

channels = test_df.sample(
    n=10,
    random_state=42
).copy()

channel_prediction = model.predict(channels[features])

# ==========================================
# 5. CHANNEL CLASSIFICATION
# ==========================================

status = []

for prediction in channel_prediction:

    prediction = str(prediction).upper()

    if prediction in ["FREE", "0"]:
        status.append("FREE")
    else:
        status.append("BUSY")

# ==========================================
# 6. FIND FREE CHANNELS
# ==========================================

free_channels = [
    i + 1 for i, s in enumerate(status)
    if s == "FREE"
]

busy_channels = [
    i + 1 for i, s in enumerate(status)
    if s == "BUSY"
]

# ==========================================
# 7. ALLOCATE CHANNEL
# ==========================================

if free_channels:
    allocated_channel = free_channels[0]
else:
    allocated_channel = None

# ==========================================
# 8. TERMINAL OUTPUT
# ==========================================

print("\n==========================================")
print("AI-BASED INTELLIGENT SPECTRUM SYSTEM")
print("==========================================")

print("\nRandom Forest Test Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\n------------------------------------------")
print("CHANNEL STATUS")
print("------------------------------------------")

for i, s in enumerate(status, start=1):

    if i == allocated_channel:
        print(f"Channel {i:2d} : {s} -> ALLOCATED")
    else:
        print(f"Channel {i:2d} : {s}")

print("\n------------------------------------------")
print("SPECTRUM SUMMARY")
print("------------------------------------------")

print(f"Total Channels : {len(status)}")
print(f"FREE Channels  : {len(free_channels)}")
print(f"BUSY Channels  : {len(busy_channels)}")

print("\nFree Channels:")
print(free_channels)

print("\nBusy Channels:")
print(busy_channels)

print("\n------------------------------------------")
print("DYNAMIC ALLOCATION")
print("------------------------------------------")

if allocated_channel:
    print(f"Selected Channel : Channel {allocated_channel}")
    print("Allocation Status : SUCCESS")
else:
    print("No FREE channel available.")
    print("Allocation Status : FAILED")

# ==========================================
# 9. DASHBOARD
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
    bars[allocated_channel - 1].set_linewidth(5)

plt.xticks(range(1, 11))
plt.yticks([0, 1], ["FREE", "BUSY"])

plt.xlabel("Channel Number")
plt.ylabel("Channel Status")
plt.title("AI-Based Intelligent Spectrum Sensing and Dynamic Allocation")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "../dataset/final_system_dashboard.png",
    dpi=300
)

plt.show()

print("\n==========================================")
print("FINAL SYSTEM COMPLETE")
print("==========================================")
print("Dashboard saved:")
print("final_system_dashboard.png")