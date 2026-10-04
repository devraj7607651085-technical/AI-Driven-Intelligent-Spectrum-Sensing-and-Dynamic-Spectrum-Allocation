import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# PATH
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

train_path = os.path.join(
    BASE_DIR,
    "dataset",
    "realistic_train_dataset.csv"
)

test_path = os.path.join(
    BASE_DIR,
    "dataset",
    "realistic_test_dataset.csv"
)


# ==========================================
# LOAD DATA
# ==========================================

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("==========================================")
print("OPTIMIZED RANDOM FOREST")
print("==========================================")

print("Training Shape :", train_df.shape)
print("Testing Shape  :", test_df.shape)


# ==========================================
# FEATURES
# ==========================================

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
# OPTIMIZED RANDOM FOREST
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


# ==========================================
# TRAIN
# ==========================================

print("\nTraining Optimized Random Forest...")

model.fit(
    X_train,
    y_train
)


# ==========================================
# OVERALL PREDICTION
# ==========================================

prediction = model.predict(
    X_test
)


# ==========================================
# OVERALL METRICS
# ==========================================

accuracy = accuracy_score(
    y_test,
    prediction
)

precision = precision_score(
    y_test,
    prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    prediction,
    zero_division=0
)


print("\n==========================================")
print("OVERALL RESULTS")
print("==========================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ==========================================
# SNR-WISE EVALUATION
# ==========================================

snr_values = sorted(
    test_df["SNR_column"].unique()
)

results = []


for snr in snr_values:

    mask = test_df["SNR_column"] == snr

    X_snr = test_df.loc[
        mask,
        features
    ]

    y_snr = test_df.loc[
        mask,
        "Label"
    ]

    prediction_snr = model.predict(
        X_snr
    )

    accuracy_snr = accuracy_score(
        y_snr,
        prediction_snr
    )

    precision_snr = precision_score(
        y_snr,
        prediction_snr,
        zero_division=0
    )

    recall_snr = recall_score(
        y_snr,
        prediction_snr,
        zero_division=0
    )

    f1_snr = f1_score(
        y_snr,
        prediction_snr,
        zero_division=0
    )

    results.append([
        snr,
        accuracy_snr,
        precision_snr,
        recall_snr,
        f1_snr
    ])


# ==========================================
# RESULTS TABLE
# ==========================================

results_df = pd.DataFrame(

    results,

    columns=[
        "SNR",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score"
    ]
)


print("\n==========================================")
print("SNR-WISE OPTIMIZED RF RESULTS")
print("==========================================")

print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# SAVE RESULTS
# ==========================================

results_path = os.path.join(
    BASE_DIR,
    "dataset",
    "optimized_rf_snr_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ==========================================
# ACCURACY GRAPH
# ==========================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    results_df["SNR"],
    results_df["Accuracy"],
    marker="o"
)

plt.title(
    "Optimized Random Forest Accuracy vs SNR"
)

plt.xlabel(
    "SNR (dB)"
)

plt.ylabel(
    "Accuracy"
)

plt.ylim(
    0,
    1.05
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "optimized_rf_accuracy.png"
    )
)

plt.show()


# ==========================================
# PRECISION GRAPH
# ==========================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    results_df["SNR"],
    results_df["Precision"],
    marker="o"
)

plt.title(
    "Optimized Random Forest Precision vs SNR"
)

plt.xlabel(
    "SNR (dB)"
)

plt.ylabel(
    "Precision"
)

plt.ylim(
    0,
    1.05
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "optimized_rf_precision.png"
    )
)

plt.show()


# ==========================================
# RECALL GRAPH
# ==========================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    results_df["SNR"],
    results_df["Recall"],
    marker="o"
)

plt.title(
    "Optimized Random Forest Recall vs SNR"
)

plt.xlabel(
    "SNR (dB)"
)

plt.ylabel(
    "Recall"
)

plt.ylim(
    0,
    1.05
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "optimized_rf_recall.png"
    )
)

plt.show()


# ==========================================
# F1 GRAPH
# ==========================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    results_df["SNR"],
    results_df["F1_Score"],
    marker="o"
)

plt.title(
    "Optimized Random Forest F1 Score vs SNR"
)

plt.xlabel(
    "SNR (dB)"
)

plt.ylabel(
    "F1 Score"
)

plt.ylim(
    0,
    1.05
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        BASE_DIR,
        "optimized_rf_f1.png"
    )
)

plt.show()


print("\n==========================================")
print("OPTIMIZATION COMPLETED")
print("==========================================")

print("\nResults saved:")
print("optimized_rf_snr_results.csv")

print("\nGraphs saved:")
print("optimized_rf_accuracy.png")
print("optimized_rf_precision.png")
print("optimized_rf_recall.png")
print("optimized_rf_f1.png")