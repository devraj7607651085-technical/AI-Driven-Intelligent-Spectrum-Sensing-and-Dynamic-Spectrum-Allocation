from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# ==========================================
# PATHS
# ==========================================

TRAIN_FILE = "dataset/realistic_train_dataset.csv"
TEST_FILE = "dataset/realistic_test_dataset.csv"
SNR_FILE = "dataset/snr_wise_results.csv"

# ==========================================
# LOAD DATA
# ==========================================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
snr_df = pd.read_csv(SNR_FILE)

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
# TRAIN RANDOM FOREST
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

test_prediction = model.predict(X_test)
rf_accuracy = accuracy_score(y_test, test_prediction)

# ==========================================
# HELPER
# ==========================================

def convert_status(prediction):
    value = str(prediction).upper()

    if value in ["FREE", "0"]:
        return "FREE"

    return "BUSY"


# ==========================================
# MAIN PAGE
# ==========================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        accuracy=round(rf_accuracy * 100, 2)
    )


# ==========================================
# DASHBOARD DATA
# ==========================================

@app.route("/api/dashboard")
def dashboard():

    try:
        snr = float(request.args.get("snr", 0))
    except:
        snr = 0

    # --------------------------------------
    # Select channels around requested SNR
    # --------------------------------------

    available = test_df[
        test_df["SNR_column"] == snr
    ].copy()

    # If exact SNR is unavailable
    if len(available) < 10:
        available = test_df.copy()

    channels = available.sample(
        n=min(10, len(available)),
        random_state=42
    ).copy()

    predictions = model.predict(channels[features])

    channel_data = []
    free_channels = []
    busy_channels = []

    for i, prediction in enumerate(predictions, start=1):

        status = convert_status(prediction)

        channel_data.append({
            "channel": i,
            "status": status
        })

        if status == "FREE":
            free_channels.append(i)
        else:
            busy_channels.append(i)

    # --------------------------------------
    # Dynamic allocation
    # --------------------------------------

    allocated_channel = (
        free_channels[0]
        if free_channels
        else None
    )

    # --------------------------------------
    # SNR performance
    # --------------------------------------

    snr_rows = snr_df.copy()

    performance = {
        "snr": snr_rows["SNR"].tolist(),
        "accuracy": snr_rows["Accuracy"].tolist(),
        "precision": snr_rows["Precision"].tolist(),
        "recall": snr_rows["Recall"].tolist(),
        "f1": snr_rows["F1_Score"].tolist()
    }

    # --------------------------------------
    # Traditional ED comparison
    # --------------------------------------

    traditional_ed = [
        0.4900,
        0.5000,
        0.5450,
        0.6750,
        0.9250,
        0.9500,
        0.9650,
        0.9450,
        0.9650
    ]

    rf_accuracy_snr = snr_rows["Accuracy"].tolist()

    comparison = {
        "snr": snr_rows["SNR"].tolist(),
        "traditional_ed": traditional_ed,
        "random_forest": rf_accuracy_snr
    }

    # --------------------------------------
    # Signal visualization
    # --------------------------------------

    Fs = 1000
    duration = 0.1

    t = np.arange(
        0,
        duration,
        1 / Fs
    )

    # Synthetic communication signal
    signal = np.sin(
        2 * np.pi * 50 * t
    )

    # Generate noise according to SNR
    signal_power = np.mean(signal ** 2)

    noise_power = (
        signal_power /
        (10 ** (snr / 10))
    )

    noise = np.random.normal(
        0,
        np.sqrt(noise_power),
        len(signal)
    )

    received_signal = signal + noise

    # --------------------------------------
    # Energy detection
    # --------------------------------------

    energy = np.sum(
        received_signal ** 2
    )

    noise_energy = np.sum(
        noise ** 2
    )

    threshold = noise_energy * 1.2

    ed_status = (
        "BUSY"
        if energy > threshold
        else "FREE"
    )

    # Send only a limited number of points
    # for smooth browser rendering

    signal_points = 300

    signal_data = {
        "time": t[:signal_points].tolist(),
        "original": signal[:signal_points].tolist(),
        "received": received_signal[:signal_points].tolist()
    }

    return jsonify({
        "accuracy": round(rf_accuracy * 100, 2),

        "channels": channel_data,

        "free_channels": free_channels,

        "busy_channels": busy_channels,

        "allocated_channel": allocated_channel,

        "total_channels": len(channel_data),

        "energy_detection": {
            "energy": round(float(energy), 4),
            "threshold": round(float(threshold), 4),
            "status": ed_status
        },

        "performance": performance,

        "comparison": comparison,

        "signal": signal_data
    })


if __name__ == "__main__":
    app.run(
        debug=True
    )