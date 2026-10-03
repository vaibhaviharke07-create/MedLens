import pandas as pd

# Load CDSCO dataset
df = pd.read_csv("data/cdsco_nsq.csv")

print("CDSCO dataset loaded successfully!")
print("Total records:", len(df))

# Test batch number
batch_to_check = "TG261185"

# Search batch
result = df[
    df["batch_no"].astype(str).str.strip().str.upper()
    == batch_to_check.strip().upper()
]

if not result.empty:
    print("\n⚠️ MATCH FOUND")
    print(result.to_string(index=False))
else:
    print("\n✅ No matching batch found.")