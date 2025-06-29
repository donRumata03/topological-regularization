import pandas as pd, glob, os, seaborn as sns, matplotlib.pyplot as plt

records = []
for cfg in range(1, 6):
    for f in glob.glob(f"results/cfg{cfg}/log_*.csv"):
        df = pd.read_csv(f)
        acc = df["acc"].iloc[-1] * 100
        records.append({"cfg": cfg, "acc": acc,
                        "file": os.path.basename(f)})

df = pd.DataFrame(records)
summary = df.drop("file", axis=1).groupby("cfg").agg(["mean", "std"]).round(2)
print(summary["acc"])
