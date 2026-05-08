"""
feature_engineering.py
======================
Script untuk melakukan feature engineering pada dataset sales_data.csv.
Menambahkan fitur waktu dan fitur turunan, lalu menyimpan ke sales_data_enriched.csv.
"""

import pandas as pd
import numpy as np

# ─── 1. LOAD DATASET ────────────────────────────────────────────────────────
print("=" * 50)
print("  FEATURE ENGINEERING — DEMAND FORECASTING")
print("=" * 50)
print("\n📂 Memuat dataset sales_data.csv ...")
df = pd.read_csv("sales_data.csv")
print(f"✅ Dataset berhasil dimuat. Shape awal: {df.shape}")

# ─── 2. KONVERSI KOLOM DATE ──────────────────────────────────────────────────
print("\n🗓️  Mengkonversi kolom 'Date' ke tipe datetime ...")
df["Date"] = pd.to_datetime(df["Date"])
print(f"✅ Kolom 'Date' berhasil dikonversi. Rentang: {df['Date'].min().date()} s/d {df['Date'].max().date()}")

# ─── 3. EXTRACT FITUR WAKTU ──────────────────────────────────────────────────
print("\n🔧 Mengekstrak fitur waktu dari kolom 'Date' ...")

df["Year"]      = df["Date"].dt.year
df["Month"]     = df["Date"].dt.month
df["Week"]      = df["Date"].dt.isocalendar().week.astype(int)
df["DayOfWeek"] = df["Date"].dt.dayofweek   # 0=Senin, 6=Minggu
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
df["Quarter"]   = df["Date"].dt.quarter

fitur_waktu = ["Year", "Month", "Week", "DayOfWeek", "IsWeekend", "Quarter"]
print(f"✅ Fitur waktu berhasil ditambahkan: {fitur_waktu}")

# ─── 4. HITUNG FITUR TURUNAN ─────────────────────────────────────────────────
print("\n📊 Menghitung fitur turunan ...")

# Revenue = penjualan * harga
df["Revenue"] = df["Units Sold"] * df["Price"]

# Dampak diskon terhadap harga
df["Discount_Impact"] = df["Discount"] * df["Price"] / 100

# Selisih harga vs kompetitor
df["Price_Diff"] = df["Price"] - df["Competitor Pricing"]

# Tingkat perputaran stok (+1 untuk menghindari pembagian dengan nol)
df["Stock_Turnover"] = df["Units Sold"] / (df["Inventory Level"] + 1)

# Flag apakah perlu restock (stok < unit terjual)
df["Restock_Needed"] = (df["Inventory Level"] < df["Units Sold"]).astype(int)

fitur_turunan = ["Revenue", "Discount_Impact", "Price_Diff", "Stock_Turnover", "Restock_Needed"]
print(f"✅ Fitur turunan berhasil ditambahkan: {fitur_turunan}")

# ─── 5. RINGKASAN HASIL ──────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  RINGKASAN HASIL FEATURE ENGINEERING")
print("=" * 50)
print(f"\n📐 Shape dataset setelah enrichment : {df.shape}")
print(f"\n📋 Daftar kolom baru yang ditambahkan:")
kolom_baru = fitur_waktu + fitur_turunan
for i, kol in enumerate(kolom_baru, 1):
    print(f"   {i:2d}. {kol}")

print(f"\n👀 Preview 5 baris pertama (kolom baru):")
print(df[["Date"] + kolom_baru].head())

# ─── 6. SIMPAN KE FILE ───────────────────────────────────────────────────────
print("\n💾 Menyimpan dataset ke 'sales_data_enriched.csv' ...")
df.to_csv("sales_data_enriched.csv", index=False)
print("✅ Dataset berhasil disimpan ke 'sales_data_enriched.csv'")
print(f"   Total baris  : {len(df):,}")
print(f"   Total kolom  : {len(df.columns)}")
print("\n🎉 Feature Engineering selesai!")
