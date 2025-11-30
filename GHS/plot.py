import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')  # cleaner style

df = pd.read_csv("results.csv")
df = df.sort_values(by="Nodes")

# Convert numeric columns
df["TimeTaken"] = df["TimeTaken"].astype(float)
df["TotalMessages"] = df["TotalMessages"].astype(int)
df["TotalDataSent"] = df["TotalDataSent"].astype(int)

# ---------- Plot 1: Nodes vs Time Taken ----------
plt.figure(figsize=(8,5))
plt.plot(df["Nodes"], df["TimeTaken"], marker="o", linewidth=2)
plt.xlabel("Nodes")
plt.ylabel("Time Taken (seconds)")
plt.title("Nodes vs Time Taken")
plt.tight_layout()
plt.savefig("plot_time_taken.png")
plt.show()

# ---------- Plot 2: Nodes vs Total Messages ----------
plt.figure(figsize=(8,5))
plt.plot(df["Nodes"], df["TotalMessages"], marker="s", linewidth=2)
plt.xlabel("Nodes")
plt.ylabel("Total Messages")
plt.title("Nodes vs Total Messages")
plt.tight_layout()
plt.savefig("plot_messages.png")
plt.show()

# ---------- Plot 3: Nodes vs Total Data Sent ----------
plt.figure(figsize=(8,5))
plt.plot(df["Nodes"], df["TotalDataSent"], marker="^", linewidth=2)
plt.xlabel("Nodes")
plt.ylabel("Total Data Sent (bytes)")
plt.title("Nodes vs Total Data Sent")
plt.tight_layout()
plt.savefig("plot_data_sent.png")
plt.show()