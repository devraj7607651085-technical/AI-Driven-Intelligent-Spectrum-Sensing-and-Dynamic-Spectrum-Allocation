import pandas as pd
import matplotlib.pyplot as plt

# Load SNR-wise results
df = pd.read_csv("../dataset/snr_wise_results.csv")

# Display results
print("\n==========================================")
print("SNR-WISE PERFORMANCE ANALYSIS")
print("==========================================")
print(df.to_string(index=False))

# Plot performance metrics
plt.figure(figsize=(10, 6))

plt.plot(df["SNR"], df["Accuracy"], marker="o", label="Accuracy")
plt.plot(df["SNR"], df["Precision"], marker="s", label="Precision")
plt.plot(df["SNR"], df["Recall"], marker="^", label="Recall")
plt.plot(df["SNR"], df["F1_Score"], marker="d", label="F1-Score")

plt.xlabel("SNR (dB)")
plt.ylabel("Performance")
plt.title("Random Forest Performance vs SNR")

plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()

plt.tight_layout()

# Save graph
plt.savefig("../dataset/RF_performance_vs_SNR.png", dpi=300)

plt.show()

print("\nGraph saved as:")
print("RF_performance_vs_SNR.png")