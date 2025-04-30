import pandas as pd, matplotlib.pyplot as plt, argparse
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--out", default="mse_bar.png")
    args=ap.parse_args()
    df=pd.read_csv(args.csv)
    plt.figure(figsize=(5,3))
    plt.bar(df['Model'], df['MSE'])
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
if __name__=='__main__': main()
