import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import Counter

CSV = "perilaku_mahasiswa.csv"
OUT = "hasil"
os.makedirs(OUT, exist_ok=True)

# ================= TAHAP 0 =================
def tahap0_eksplorasi(df):
    print("=" * 60)
    print("TAHAP 0 - EKSPLORASI AWAL")
    print("=" * 60)
    print("Shape awal:", df.shape)
    print("Kolom:", list(df.columns))
    print(df.head(5).to_string())
    print("\nTipe data:\n", df.dtypes)
    print("\nMissing per kolom:\n", df.isna().sum())
    print("Duplikat:", df.duplicated().sum())
    return df

# ================= TAHAP 1 =================
def tahap1_missing(df):
    print("\n" + "=" * 60)
    print("TAHAP 1 - MISSING VALUE")
    print("=" * 60)
    print("Missing sebelum:", int(df.isna().sum().sum()))
    for c in df.columns:
        if df[c].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "Tidak Diketahui")
    print("Missing sesudah:", int(df.isna().sum().sum()))
    return df

# ================= TAHAP 2 =================
def tahap2_duplikat(df):
    print("\n" + "=" * 60)
    print("TAHAP 2 - DUPLIKAT")
    print("=" * 60)
    print("Sebelum:", len(df))
    df = df.drop_duplicates().reset_index(drop=True)
    print("Sesudah:", len(df))
    return df

# ================= TAHAP 3 =================
def tahap3_harmonisasi(df):
    print("\n" + "=" * 60)
    print("TAHAP 3 - HARMONISASI LABEL")
    print("=" * 60)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.title()
            df[c] = df[c].replace({"Ya": "Ya", "Tidak": "Tidak", "L": "Laki-Laki", "P": "Perempuan"})
    print("Kolom sesudah:", list(df.columns))
    return df

# ================= TAHAP 4 =================
def tahap4_tipe_data(df):
    print("\n" + "=" * 60)
    print("TAHAP 4 - TIPE DATA")
    print("=" * 60)
    for c in df.columns:
        if "usia" in c or "umur" in c or "semester" in c or "ipk" in c or "sks" in c or "jam" in c or "nilai" in c:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    print(df.dtypes)
    return df

# ================= TAHAP 5 =================
def tahap5_outlier(df):
    print("\n" + "=" * 60)
    print("TAHAP 5 - OUTLIER (IQR)")
    print("=" * 60)
    num = df.select_dtypes(include=[np.number]).columns
    for c in num:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        sebelum = len(df)
        df = df[(df[c] >= lo) & (df[c] <= hi)]
        print(f"{c}: {sebelum} -> {len(df)}")
    return df.reset_index(drop=True)

# ================= TAHAP 6 =================
def tahap6_konsistensi(df):
    print("\n" + "=" * 60)
    print("TAHAP 6 - KONSISTENSI")
    print("=" * 60)
    if "usia" in df.columns:
        df = df[(df["usia"] >= 15) & (df["usia"] <= 40)]
    if "semester" in df.columns:
        df = df[(df["semester"] >= 1) & (df["semester"] <= 14)]
    if "ipk" in df.columns:
        df = df[(df["ipk"] >= 0) & (df["ipk"] <= 4)]
    print("Shape akhir bersih:", df.shape)
    return df.reset_index(drop=True)

# ================= TAHAP 7 =================
def tahap7_visualisasi(df):
    print("\n" + "=" * 60)
    print("TAHAP 7 - VISUALISASI")
    print("=" * 60)
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=["object"]).columns.tolist()
    if len(num) >= 1:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[num[0]], kde=True)
        plt.title(f"Distribusi {num[0]}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "hist.png"))
        plt.close()
    if len(cat) >= 1:
        plt.figure(figsize=(8, 5))
        df[cat[0]].value_counts().head(10).plot(kind="bar")
        plt.title(f"Distribusi {cat[0]}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "bar.png"))
        plt.close()
    if len(num) >= 2:
        plt.figure(figsize=(7, 5))
        sns.heatmap(df[num].corr(numeric_only=True), annot=True, cmap="coolwarm")
        plt.title("Korelasi Numerik")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "heatmap.png"))
        plt.close()
    print("Grafik tersimpan di folder:", OUT)

# ================= TAHAP 8 =================
def tahap8_fp_growth(df):
    print("\n" + "=" * 60)
    print("TAHAP 8 - FP-GROWTH")
    print("=" * 60)
    # Bikin transaksi dari kolom kategorikal
    cat = df.select_dtypes(include=["object"]).columns.tolist()
    if len(cat) == 0:
        cat = df.columns.tolist()[:3]
    transaksi = []
    for _, row in df.iterrows():
        items = []
        for c in cat[:4]:
            items.append(f"{c}={row[c]}")
        transaksi.append(set(items))
    # Frequent 1-itemset
    min_support = 0.1
    min_count = int(len(transaksi) * min_support)
    cnt = Counter()
    for t in transaksi:
        for item in t:
            cnt[item] += 1
    frequent1 = {k: v for k, v in cnt.items() if v >= min_count}
    print(f"Frequent 1-itemset (support >= {min_support}): {len(frequent1)}")
    # Frequent 2-itemset
    cnt2 = Counter()
    for t in transaksi:
        items = [x for x in t if x in frequent1]
        for a, b in combinations(sorted(items), 2):
            cnt2[(a, b)] += 1
    frequent2 = {k: v for k, v in cnt2.items() if v >= min_count}
    print(f"Frequent 2-itemset: {len(frequent2)}")
    # Aturan asosiasi
    aturan = []
    for (a, b), c in frequent2.items():
        conf_ab = c / cnt[a] if cnt[a] else 0
        conf_ba = c / cnt[b] if cnt[b] else 0
        sup = c / len(transaksi)
        if conf_ab >= 0.5:
            aturan.append((a, b, sup, conf_ab))
        if conf_ba >= 0.5:
            aturan.append((b, a, sup, conf_ba))
    aturan = sorted(aturan, key=lambda x: x[3], reverse=True)[:20]
    print("\nTop 20 Aturan:")
    for a, b, sup, conf in aturan:
        print(f"{a} => {b} | support={sup:.3f} confidence={conf:.3f}")
    # Simpan
    pd.DataFrame(aturan, columns=["Antecedent", "Consequent", "Support", "Confidence"]).to_csv(os.path.join(OUT, "aturan_fp_growth.csv"), index=False)
    pd.DataFrame([{"Item": k, "Count": v, "Support": v/len(transaksi)} for k, v in frequent1.items()]).sort_values("Count", ascending=False).to_csv(os.path.join(OUT, "frequent_itemset.csv"), index=False)
    return aturan, frequent1

# ================= TAHAP 9 =================
def tahap9_kesimpulan(df, aturan, frequent):
    print("\n" + "=" * 60)
    print("TAHAP 9 - KESIMPULAN")
    print("=" * 60)
    print(f"Data akhir: {len(df)} baris, {len(df.columns)} kolom")
    print(f"Jumlah frequent item: {len(frequent)}")
    print(f"Jumlah aturan: {len(aturan)}")
    if aturan:
        print(f"Aturan terkuat: {aturan[0][0]} => {aturan[0][1]} (conf={aturan[0][3]:.2f})")
    print("Kesimpulan: Data sudah bersih, siap dipakai untuk analisis FP-Growth.")

# ================= MAIN =================
def main():
    print("=" * 60)
    print("PRAKTIKUM DATA MINING - DATA PERILAKU MAHASISWA")
    print("ADVANCED DATA CLEANING + FP-GROWTH")
    print("=" * 60)

    if not os.path.exists(CSV):
        raise FileNotFoundError(
            "File '" + CSV + "' tidak ditemukan. Letakkan file CSV "
            "satu folder dengan script Python."
        )

    df = pd.read_csv(CSV)
    print("Dataset berhasil dibaca:", CSV)
    print("Jumlah data awal:", len(df))

    df = tahap0_eksplorasi(df)
    df = tahap1_missing(df)
    df = tahap2_duplikat(df)
    df = tahap3_harmonisasi(df)
    df = tahap4_tipe_data(df)
    df = tahap5_outlier(df)
    df = tahap6_konsistensi(df)

    file_bersih = os.path.join(OUT, "perilaku_mahasiswa_bersih.csv")
    df.to_csv(file_bersih, index=False)
    print("\nData bersih disimpan:", file_bersih)

    tahap7_visualisasi(df)
    aturan, frequent = tahap8_fp_growth(df)
    tahap9_kesimpulan(df, aturan, frequent)

    print("\nSelesai. Semua hasil tersimpan di folder:", OUT)

if __name__ == "__main__":
    main()
