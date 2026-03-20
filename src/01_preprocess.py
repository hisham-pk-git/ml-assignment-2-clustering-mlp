import pandas as pd
from sklearn.preprocessing import StandardScaler

SEED = 42

INPUT_CSV = "data/GasProperties.csv"
OUTPUT_STD = "outputs/StdGasProperties.csv"

FEATURES = ["T", "P", "TC", "SV"]  # NOT included Idx for clustering

def main():
    df = pd.read_csv(INPUT_CSV)

    X = df[FEATURES].copy()

    print("=== Before standardization (mean/std) ===")
    print(X.mean())
    print(X.std(ddof=0))

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    df_std = pd.DataFrame(X_std, columns=FEATURES)

    print("\n=== After standardization (mean/std) ===")
    print(df_std.mean())
    print(df_std.std(ddof=0))

    df_std.to_csv(OUTPUT_STD, index=False)
    print(f"\nSaved: {OUTPUT_STD}")

if __name__ == "__main__":
    main()