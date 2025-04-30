import argparse, csv, glob, re, os, pathlib
def scan(root):
    rows=[]
    for fp in glob.glob(f"{root}/**/result.txt", recursive=True):
        txt=open(fp).read()
        mse=float(re.search(r"MSE=([0-9.]+)",txt).group(1))
        mae=float(re.search(r"MAE=([0-9.]+)",txt).group(1))
        model=pathlib.Path(fp).parent.name
        rows.append((model,mse,mae))
    return sorted(rows, key=lambda x:x[1])
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("csv_out")
    ap.add_argument("--latex")
    args=ap.parse_args()
    rows=scan(args.root)
    with open(args.csv_out,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["Model","MSE","MAE"]); w.writerows(rows)
    if args.latex:
        with open(args.latex,"w") as f:
            f.write("\begin{tabular}{lcc}\toprule\n")
            f.write("Model & MSE & MAE\\\\\midrule\n")
            for m,mse,mae in rows:
                f.write(f"{m} & {mse:.4f} & {mae:.4f}\\\\\n")
            f.write("\bottomrule\n\end{tabular}")
if __name__=='__main__': main()
