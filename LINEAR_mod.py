"""
LINEAR_mod.py - Model Linear Regression untuk prediksi Demand (Kategori: Clothing)
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns, joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 50)
print("  MODEL LINEAR REGRESSION — DEMAND FORECASTING")
print("  (Kategori: Clothing)")
print("=" * 50)

# ─── 1. LOAD DATA ───────────────────────────────────────────────────────────
if os.path.exists("sales_data_enriched.csv"):
    print("\n📂 Memuat 'sales_data_enriched.csv' ...")
    df = pd.read_csv("sales_data_enriched.csv", parse_dates=["Date"])
else:
    print("\n⚠️  Fallback ke 'sales_data.csv' — membuat fitur waktu ...")
    df = pd.read_csv("sales_data.csv", parse_dates=["Date"])
    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["Week"]      = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["Quarter"]   = df["Date"].dt.quarter
print(f"✅ Shape: {df.shape}")

# ─── 2. FILTER CLOTHING ─────────────────────────────────────────────────────
df_cloth = df[df["Category"] == "Clothing"].copy()
print(f"\n🔍 Data Clothing: {len(df_cloth):,} baris")

# ─── 3. DEFINISI FITUR & TARGET ─────────────────────────────────────────────
kolom_kat = ["Seasonality", "Weather Condition", "Region", "Store ID"]
kolom_num = ["Month","Year","Week","DayOfWeek","IsWeekend",
             "Price","Discount","Promotion","Inventory Level",
             "Competitor Pricing","Epidemic"]
target = "Demand"

# ─── 4. OUTLIER REMOVAL IQR ─────────────────────────────────────────────────
print("\n🧹 Outlier removal IQR pada 'Demand' ...")
sebelum = len(df_cloth)
Q1, Q3  = df_cloth[target].quantile(0.25), df_cloth[target].quantile(0.75)
IQR     = Q3 - Q1
bawah, atas = Q1 - 1.5*IQR, Q3 + 1.5*IQR
df_clean = df_cloth[(df_cloth[target] >= bawah) & (df_cloth[target] <= atas)].copy()
print(f"   SEBELUM: {sebelum:,}  |  SESUDAH: {len(df_clean):,}  |  Dihapus: {sebelum-len(df_clean):,}")

# Boxplot
fig, ax = plt.subplots(1, 2, figsize=(12,5))
ax[0].boxplot(df_cloth[target], patch_artist=True,
              boxprops=dict(facecolor="#4A90D9"), medianprops=dict(color="red",linewidth=2))
ax[0].set_title("Sebelum Outlier Removal"); ax[0].set_ylabel("Demand")
ax[1].boxplot(df_clean[target], patch_artist=True,
              boxprops=dict(facecolor="#2ecc71"), medianprops=dict(color="red",linewidth=2))
ax[1].set_title("Sesudah Outlier Removal (IQR)"); ax[1].set_ylabel("Demand")
plt.suptitle("Outlier Removal — Linear Regression (Clothing)", fontweight="bold")
plt.tight_layout(); plt.savefig("outlier_linear.png", dpi=150, bbox_inches="tight"); plt.close()
print("✅ outlier_linear.png disimpan")

X = df_clean[kolom_kat + kolom_num]
y = df_clean[target]

# ─── 5. PIPELINE + TRAIN ────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), kolom_kat),
    ("num", StandardScaler(), kolom_num),
])
pipeline = Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n🚀 Training: {len(X_train):,} | Test: {len(X_test):,}")
pipeline.fit(X_train, y_train)

# ─── 6. EVALUASI ────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
r2   = r2_score(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
mask = y_test != 0
mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

print("\n" + "=" * 44)
print("  EVALUASI MODEL LINEAR REGRESSION")
print("  (Kategori: Clothing)")
print("=" * 44)
print(f"  R² Score           : {r2:.4f}")
print(f"  MSE                : {mse:.2f}")
print(f"  RMSE               : {rmse:.2f}")
print(f"  MAE                : {mae:.2f}")
print(f"  MAPE               : {mape:.2f}%")
print("=" * 44)

# ─── 7. VISUALISASI ─────────────────────────────────────────────────────────
# Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.4, color="#3498db", edgecolors="none")
v_min = min(y_test.min(), y_pred.min()); v_max = max(y_test.max(), y_pred.max())
plt.plot([v_min,v_max],[v_min,v_max],"r--",linewidth=2,label="Prediksi Sempurna")
plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.title("Actual vs Predicted — Linear Regression (Clothing)", fontweight="bold")
plt.legend(); plt.tight_layout()
plt.savefig("actual_vs_pred_linear.png", dpi=150, bbox_inches="tight"); plt.close()

# Residual plot
residuals = np.array(y_test) - y_pred
plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuals, alpha=0.4, color="#e74c3c", edgecolors="none")
plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
plt.xlabel("Predicted"); plt.ylabel("Residuals")
plt.title("Residual Plot — Linear Regression", fontweight="bold")
plt.tight_layout(); plt.savefig("residual_linear.png", dpi=150, bbox_inches="tight"); plt.close()

# Distribusi residuals
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=40, kde=True, color="#9b59b6")
plt.xlabel("Residuals"); plt.ylabel("Frekuensi")
plt.title("Distribusi Residuals — Linear Regression", fontweight="bold")
plt.tight_layout(); plt.savefig("residual_dist_linear.png", dpi=150, bbox_inches="tight"); plt.close()
print("✅ Semua visualisasi disimpan: actual_vs_pred_linear.png, residual_linear.png, residual_dist_linear.png")

# ─── 8. CONFIDENCE INTERVAL 95% ────────────────────────────────────────────
std_err  = np.std(residuals)
ci_lower = y_pred - 1.96 * std_err
ci_upper = y_pred + 1.96 * std_err
print("\n📐 Contoh 5 prediksi pertama + CI 95%:")
print(f"  {'No':>3} | {'Actual':>8} | {'Pred':>10} | {'CI Lower':>10} | {'CI Upper':>10}")
for i in range(5):
    print(f"  {i+1:>3} | {list(y_test)[i]:>8.2f} | {y_pred[i]:>10.2f} | {ci_lower[i]:>10.2f} | {ci_upper[i]:>10.2f}")

# ─── 9. YoY ANALYSIS ────────────────────────────────────────────────────────
print("\n📅 Year-on-Year Analysis — Rata-rata Demand per Tahun (Clothing):")
yoy = df_clean.groupby("Year")["Demand"].mean().reset_index()
yoy.columns = ["Year","Avg Demand"]
yoy["Growth (%)"] = yoy["Avg Demand"].pct_change() * 100
print(yoy.to_string(index=False))

# ─── 10. SAVE MODEL ─────────────────────────────────────────────────────────
joblib.dump(pipeline, "linear_model.pkl")
print("\n✅ Model Linear Regression berhasil disimpan ke linear_model.pkl")
print("🎉 LINEAR_mod.py selesai!")
