"""
Convert MATLAB .mat data files used in BVAR_ examples to CSV format.
Requires: scipy, numpy, pandas, openpyxl
"""
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

MAT_DIR = r"C:\Users\326326\Documents\Github\BVAR_\examples\BVAR tutorial"
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)
skip = {"__header__", "__version__", "__globals__"}


def extract_varnames(d, key="varnames"):
    raw = d[key]
    return [str(v.flat[0]).strip() if hasattr(v, 'flat') else str(v).strip()
            for v in raw.flat]


def save_df(df, filename):
    path = os.path.join(OUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Saved {filename}: {df.shape}")


# ── 1. DataGK.mat ──
print("Converting DataGK.mat...")
d = loadmat(os.path.join(MAT_DIR, "DataGK.mat"))
varnames_gk = extract_varnames(d, "varnames")
T_gk = d["T"].flatten()
num_gk = d["num"]
df_gk = pd.DataFrame(num_gk, columns=varnames_gk)
df_gk.insert(0, "T", T_gk[:len(num_gk)])
save_df(df_gk, "DataGK.csv")

# ── 2. Data.mat ──
print("Converting Data.mat...")
d = loadmat(os.path.join(MAT_DIR, "Data.mat"))
varnames_data = extract_varnames(d, "varnames")
T_data = d["T"].flatten()
cols = {}
for vn in varnames_data:
    if vn in d:
        cols[vn] = d[vn].flatten()
df_data = pd.DataFrame(cols)
df_data.insert(0, "T", T_data[:len(df_data)])
save_df(df_data, "Data.csv")

# ── 3. DataMF.mat ──
print("Converting DataMF.mat...")
d = loadmat(os.path.join(MAT_DIR, "DataMF.mat"))
varnames_mf = extract_varnames(d, "varnames")
T_mf = d["T"].flatten()
cols = {}
for vn in varnames_mf:
    if vn in d:
        cols[vn] = d[vn].flatten()
df_mf = pd.DataFrame(cols)
df_mf.insert(0, "T", T_mf[:len(df_mf)])
save_df(df_mf, "DataMF.csv")

# ── 4. DataBanks.mat ──
print("Converting DataBanks.mat...")
d = loadmat(os.path.join(MAT_DIR, "DataBanks.mat"))
lr = d["LendingRate"]
dr = d["DepositRate"]
pd.DataFrame(lr).to_csv(os.path.join(OUT_DIR, "DataBanks_LendingRate.csv"),
                         index=False, header=False)
pd.DataFrame(dr).to_csv(os.path.join(OUT_DIR, "DataBanks_DepositRate.csv"),
                         index=False, header=False)
print(f"  Saved DataBanks_LendingRate.csv: {lr.shape}")
print(f"  Saved DataBanks_DepositRate.csv: {dr.shape}")

# ── 5. DataPooling.mat ──
print("Converting DataPooling.mat...")
d = loadmat(os.path.join(MAT_DIR, "DataPooling.mat"))
pool_data = {}
for key in sorted(d.keys()):
    if key in skip:
        continue
    val = d[key]
    if isinstance(val, np.ndarray) and val.ndim <= 2 and val.dtype.kind in ('f', 'i'):
        pool_data[key] = val.flatten()
max_len = max(len(v) for v in pool_data.values())
for k in pool_data:
    if len(pool_data[k]) < max_len:
        pool_data[k] = np.pad(pool_data[k], (0, max_len - len(pool_data[k])),
                               constant_values=np.nan)
save_df(pd.DataFrame(pool_data), "DataPooling.csv")

# ── 6. DataFAVAR.mat ──
print("Converting DataFAVAR.mat...")
d = loadmat(os.path.join(MAT_DIR, "DataFAVAR.mat"))
y1_favar = d["y1"]
y2_favar = d["y2"]
try:
    varnames_y2 = extract_varnames(d, "varnames_y2")
except:
    varnames_y2 = [f"V{i+1}" for i in range(y2_favar.shape[1])]
pd.DataFrame(y1_favar, columns=["TBILL3M"]).to_csv(
    os.path.join(OUT_DIR, "DataFAVAR_y1.csv"), index=False)
pd.DataFrame(y2_favar, columns=varnames_y2).to_csv(
    os.path.join(OUT_DIR, "DataFAVAR_y2.csv"), index=False)
pd.DataFrame({"varnames_y2": varnames_y2}).to_csv(
    os.path.join(OUT_DIR, "DataFAVAR_varnames.csv"), index=False)
print(f"  Saved DataFAVAR: y1={y1_favar.shape}, y2={y2_favar.shape}")

# ── 7. DataEx.mat ──
print("Converting DataEx.mat...")
d = loadmat(os.path.join(MAT_DIR, "DataEx.mat"))
ex_data = {}
for key in sorted(d.keys()):
    if key in skip:
        continue
    val = d[key]
    if isinstance(val, np.ndarray) and val.ndim <= 2 and val.dtype.kind in ('f', 'i') and val.size > 1:
        ex_data[key] = val.flatten()
if ex_data:
    max_len = max(len(v) for v in ex_data.values())
    for k in ex_data:
        if len(ex_data[k]) < max_len:
            ex_data[k] = np.pad(ex_data[k], (0, max_len - len(ex_data[k])),
                                 constant_values=np.nan)
    save_df(pd.DataFrame(ex_data), "DataEx.csv")

# ── 8. DataCovid.mat ──
print("Converting DataCovid.mat...")
try:
    d = loadmat(os.path.join(MAT_DIR, "DataCovid.mat"))
    time_covid = d["time"].flatten()
    y_covid = d["y"]
    covid_cols = ["PAYEMS", "UNRATE", "PCE", "INDPRO", "CPIAUCSL", "PCEPILFE"]
    df_c = pd.DataFrame(y_covid, columns=covid_cols)
    df_c.insert(0, "time", time_covid[:len(y_covid)])
    save_df(df_c, "DataCovid.csv")
except Exception as e:
    print(f"  Skipped DataCovid.mat: {e}")

# ── 9. factor_data.xlsx ──
print("Converting factor_data.xlsx...")
try:
    xlsx_path = os.path.join(MAT_DIR, "factor_data.xlsx")
    df_fd = pd.read_excel(xlsx_path, sheet_name="factor_data")
    save_df(df_fd, "factor_data.csv")
except Exception as e:
    print(f"  Skipped factor_data.xlsx: {e}")

# ── 10. InstrumentGK.mat ──
print("Converting InstrumentGK.mat...")
try:
    d = loadmat(os.path.join(MAT_DIR, "InstrumentGK.mat"))
    for key in sorted(d.keys()):
        if key in skip:
            continue
        val = d[key]
        if isinstance(val, np.ndarray) and val.ndim <= 2 and val.size > 1:
            pd.DataFrame(val).to_csv(
                os.path.join(OUT_DIR, f"InstrumentGK_{key}.csv"),
                index=False, header=False)
            print(f"  Saved InstrumentGK_{key}.csv: {val.shape}")
except Exception as e:
    print(f"  Skipped InstrumentGK.mat: {e}")

# ── 11. crypto_all.xls ──
print("Converting crypto_all.xls...")
try:
    xls_path = os.path.join(MAT_DIR, "crypto_all.xls")
    for sheet_name in ["price", "high", "low"]:
        df_crypto = pd.read_excel(xls_path, sheet_name=sheet_name)
        save_df(df_crypto, f"crypto_{sheet_name}.csv")
except Exception as e:
    print(f"  Skipped crypto_all.xls: {e}")

print("\nDone! All CSV files saved to:", OUT_DIR)
