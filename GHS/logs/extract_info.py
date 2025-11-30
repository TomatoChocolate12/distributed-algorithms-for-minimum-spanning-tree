import re
import csv
import glob

# Regex patterns
patterns = {
    "Nodes": r"Nodes:\s*(\d+)",
    "TimeTaken": r"Time Taken:\s*([\deE\.\-]+)",
    "TotalMessages": r"Total Messages:\s*(\d+)",
    "TotalDataSent": r"Total Data Sent:\s*(\d+)",
    "AvgDataNode": r"Avg Data/Node:\s*(\d+)",
    "MaxLevel": r"Max Level:\s*(\d+)"
}

output_rows = []

for filename in glob.glob("*.txt"):
    with open(filename, "r") as f:
        text = f.read()

    extracted = {"File": filename}
    valid = True
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            extracted[key] = match.group(1)
        else:
            valid = False
            print(f"Warning: '{key}' not found in {filename}")
    
    if valid:
        output_rows.append(extracted)

# Write to CSV
with open("results.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
        "File", "Nodes", "TimeTaken", "TotalMessages", "TotalDataSent",
        "AvgDataNode", "MaxLevel"
    ])
    writer.writeheader()
    writer.writerows(output_rows)

print("CSV written to ghs_results.csv")