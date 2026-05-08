"""
clustering_mod.py - KMeans Clustering Produk berdasarkan karakteristik penjualan
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("=" * 50)
print("  CLUSTERING PRODUK — DEMAND FORECASTING")
print("=" * 50)

# ─── 1. LOAD DATA ───────────────────────────────────────────────────────────
print("\n📂 Memuat 'sales_data.csv' ...")
df = pd.read_csv("sales_data.csv")
print(f"✅ Shape: {df.shape}")

# ─── 2. AGREGASI PER PRODUCT ID ─────────────────────────────────────────────
print("\n🔧 Mengagregasi data per Product ID ...")
agg = df.groupby("Product ID").agg(
    avg_demand       = ("Demand",          "mean"),
    avg_price        = ("Price",           "mean"),
    total_units_sold = ("Units Sold",      "sum"),
    avg_inventory    = ("Inventory Level", "mean"),
    avg_discount     = ("Discount",        "mean"),
    promo_rate       = ("Promotion",       "mean"),
).reset_index()

# Tambahkan kolom Category (mode per Product ID)
cat_mode = df.groupby("Product ID")["Category"].agg(lambda x: x.mode()[0]).reset_index()
cat_mode.columns = ["Product ID", "Category"]
agg = agg.merge(cat_mode, on="Product ID")
print(f"✅ Agregasi selesai. Shape: {agg.shape} (1 baris per produk)")
print(agg.to_string(index=False))

# ─── 3. SCALING ─────────────────────────────────────────────────────────────
fitur_cluster = ["avg_demand","avg_price","total_units_sold","avg_inventory","avg_discount","promo_rate"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(agg[fitur_cluster])
print(f"\n✅ Fitur di-scale: {fitur_cluster}")

# ─── 4. ELBOW METHOD ────────────────────────────────────────────────────────
print("\n📉 Menjalankan Elbow Method (k=2-10) ...")
inertias = []
k_range  = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(list(k_range), inertias, "bo-", linewidth=2, markersize=8)
plt.xlabel("Jumlah Cluster (k)"); plt.ylabel("Inertia")
plt.title("Elbow Method — Pemilihan k Optimal", fontweight="bold")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("elbow_method.png", dpi=150, bbox_inches="tight"); plt.close()
print("✅ elbow_method.png disimpan")

# ─── 5. SILHOUETTE SCORE ────────────────────────────────────────────────────
print("\n📊 Menghitung Silhouette Score (k=2-10) ...")
silhouette_scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sc = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sc)
    print(f"   k={k}: Silhouette Score = {sc:.4f}")

k_optimal = list(k_range)[np.argmax(silhouette_scores)]
print(f"\n🏆 k Optimal berdasarkan Silhouette Score: k = {k_optimal}")

plt.figure(figsize=(8,5))
plt.plot(list(k_range), silhouette_scores, "rs-", linewidth=2, markersize=8)
plt.axvline(x=k_optimal, color="green", linestyle="--", label=f"k Optimal = {k_optimal}")
plt.xlabel("Jumlah Cluster (k)"); plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs k — Pemilihan Cluster Optimal", fontweight="bold")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("silhouette_score.png", dpi=150, bbox_inches="tight"); plt.close()
print("✅ silhouette_score.png disimpan")

# ─── 6. FIT KMEANS OPTIMAL ──────────────────────────────────────────────────
print(f"\n🔧 Menjalankan KMeans dengan k_optimal = {k_optimal} ...")
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
agg["Cluster"] = kmeans.fit_predict(X_scaled)
print(f"✅ Cluster berhasil ditambahkan ke DataFrame")

# ─── 7. VISUALISASI CLUSTER ─────────────────────────────────────────────────
print("\n🎨 Membuat scatter plot cluster ...")
palette = sns.color_palette("tab10", k_optimal)
plt.figure(figsize=(10,7))
for c in range(k_optimal):
    subset = agg[agg["Cluster"] == c]
    plt.scatter(subset["avg_price"], subset["avg_demand"],
                label=f"Cluster {c}", color=palette[c], s=120, edgecolors="black", linewidth=0.8)
    for _, row in subset.iterrows():
        plt.annotate(row["Product ID"],
                     (row["avg_price"], row["avg_demand"]),
                     textcoords="offset points", xytext=(6,4), fontsize=8)
plt.xlabel("Rata-rata Harga (avg_price)", fontsize=12)
plt.ylabel("Rata-rata Demand (avg_demand)", fontsize=12)
plt.title("Clustering Produk — Harga vs Demand", fontsize=14, fontweight="bold")
plt.legend(title="Cluster"); plt.tight_layout()
plt.savefig("cluster_scatter.png", dpi=150, bbox_inches="tight"); plt.close()
print("✅ cluster_scatter.png disimpan")

# ─── 8. INTERPRETASI CLUSTER OTOMATIS ───────────────────────────────────────
print("\n🧠 Interpretasi otomatis setiap cluster ...")
median_demand_global = agg["avg_demand"].median()
median_price_global  = agg["avg_price"].median()

interpretasi_map = {}
print(f"\n   Median global — Demand: {median_demand_global:.2f} | Harga: {median_price_global:.2f}")
print("\n" + "=" * 65)
print(f"  {'Cluster':^8} | {'Med.Demand':^10} | {'Med.Harga':^10} | {'Interpretasi':<30}")
print("=" * 65)

for c in range(k_optimal):
    subset = agg[agg["Cluster"] == c]
    med_d  = subset["avg_demand"].median()
    med_p  = subset["avg_price"].median()
    if   med_d >= median_demand_global and med_p <  median_price_global:
        label = "🟢 Produk Laris Ekonomis"
    elif med_d >= median_demand_global and med_p >= median_price_global:
        label = "🔵 Produk Premium Laku"
    elif med_d <  median_demand_global and med_p >= median_price_global:
        label = "🟡 Produk Premium Lambat"
    else:
        label = "🔴 Produk Kurang Diminati"
    interpretasi_map[c] = label
    print(f"  {c:^8} | {med_d:^10.2f} | {med_p:^10.2f} | {label:<30}")
print("=" * 65)

agg["Interpretasi"] = agg["Cluster"].map(interpretasi_map)
print("\n📋 Tabel produk per cluster:")
print(agg[["Product ID","Category","Cluster","Interpretasi","avg_demand","avg_price"]].to_string(index=False))

# Simpan hasil clustering
agg.to_csv("clustering_result.csv", index=False)
print("\n✅ clustering_result.csv disimpan")

# ─── 9. SAVE MODELS ─────────────────────────────────────────────────────────
joblib.dump(scaler, "cluster_scaler.pkl")
joblib.dump(kmeans, "cluster_model.pkl")
print("✅ cluster_scaler.pkl disimpan")
print("✅ cluster_model.pkl disimpan")
print("\n🎉 clustering_mod.py selesai!")
