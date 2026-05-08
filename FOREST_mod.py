"""
FOREST_mod.py - Model Random Forest Regressor untuk prediksi Demand (Semua Kategori)
"""
import os, time, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 50)
print("  MODEL RANDOM FOREST — DEMAND FORECASTING")
print("  (Semua Kategori)")
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
print(f"✅ Shape dataset: {df.shape}")

# ─── 2. DEFINISI FITUR & TARGET ─────────────────────────────────────────────
kolom_kat = ["Seasonality", "Weather Condition", "Region", "Store ID", "Category", "Product ID"]
kolom_num = ["Month","Year","Week","DayOfWeek","IsWeekend",
             "Price","Discount","Promotion","Inventory Level",
             "Competitor Pricing","Epidemic"]
target = "Demand"

# ─── 3. OUTLIER REMOVAL IQR ─────────────────────────────────────────────────
print("\n🧹 Outlier removal IQR pada 'Demand' ...")
sebelum = len(df)
Q1, Q3  = df[target].quantile(0.25), df[target].quantile(0.75)
IQR     = Q3 - Q1
bawah, atas = Q1 - 1.5*IQR, Q3 + 1.5*IQR
df_clean = df[(df[target] >= bawah) & (df[target] <= atas)].copy()
print(f"   SEBELUM: {sebelum:,}  |  SESUDAH: {len(df_clean):,}  |  Dihapus: {sebelum-len(df_clean):,}")

# Boxplot
fig, ax = plt.subplots(1, 2, figsize=(12,5))
ax[0].boxplot(df[target], patch_artist=True,
              boxprops=dict(facecolor="#4A90D9"), medianprops=dict(color="red",linewidth=2))
ax[0].set_title("Sebelum Outlier Removal"); ax[0].set_ylabel("Demand")
ax[1].boxplot(df_clean[target], patch_artist=True,
              boxprops=dict(facecolor="#2ecc71"), medianprops=dict(color="red",linewidth=2))
ax[1].set_title("Sesudah Outlier Removal (IQR)"); ax[1].set_ylabel("Demand")
plt.suptitle("Outlier Removal — Random Forest (Semua Kategori)", fontweight="bold")
plt.tight_layout(); plt.savefig("outlier_forest.png", dpi=150, bbox_inches="tight"); plt.close()
print("✅ outlier_forest.png disimpan")

X = df_clean[kolom_kat + kolom_num]
y = df_clean[target]

# ─── 4. PIPELINE ────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), kolom_kat),
    ("num", StandardScaler(), kolom_num),
])
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42)),
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n📊 Train: {len(X_train):,} | Test: {len(X_test):,}")

# ─── 5. HYPERPARAMETER TUNING ───────────────────────────────────────────────
param_grid = {
    "model__n_estimators":    [100, 200, 300],
    "model__max_depth":       [10, 20, None],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf":  [1, 2],
}
print("\n⏳ Proses training Random Forest + GridSearchCV dimulai ...")
print("⏳ Estimasi waktu: 5-15 menit tergantung spesifikasi komputer")

start_time = time.time()
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
elapsed = time.time() - start_time
print(f"\n✅ Training selesai dalam {elapsed:.1f} detik")
print(f"   Best Parameters : {grid_search.best_params_}")
print(f"   Best CV R² Score: {grid_search.best_score_:.4f}")

best_pipeline = grid_search.best_estimator_

# ─── 6. EVALUASI ────────────────────────────────────────────────────────────
y_pred = best_pipeline.predict(X_test)
r2   = r2_score(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
mask = y_test != 0
mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

print("\n" + "=" * 44)
print("  EVALUASI MODEL RANDOM FOREST")
print("  (Semua Kategori)")
print("=" * 44)
print(f"  R² Score           : {r2:.4f}")
print(f"  MSE                : {mse:.2f}")
print(f"  RMSE               : {rmse:.2f}")
print(f"  MAE                : {mae:.2f}")
print(f"  MAPE               : {mape:.2f}%")
print("=" * 44)

# ─── 7. VISUALISASI ─────────────────────────────────────────────────────────
# Feature Importance
feat_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = best_pipeline.named_steps["model"].feature_importances_
feat_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
feat_df = feat_df.sort_values("Importance", ascending=False).head(15)

plt.figure(figsize=(10,7))
sns.barplot(data=feat_df, y="Feature", x="Importance", palette="viridis")
plt.title("Top 15 Feature Importance — Random Forest", fontweight="bold", fontsize=13)
plt.xlabel("Importance"); plt.ylabel("Feature")
plt.tight_layout(); plt.savefig("feature_importance_forest.png", dpi=150, bbox_inches="tight"); plt.close()

# Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.3, color="#3498db", edgecolors="none")
v_min = min(y_test.min(), y_pred.min()); v_max = max(y_test.max(), y_pred.max())
plt.plot([v_min,v_max],[v_min,v_max],"r--",linewidth=2,label="Prediksi Sempurna")
plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.title("Actual vs Predicted — Random Forest", fontweight="bold")
plt.legend(); plt.tight_layout()
plt.savefig("actual_vs_pred_forest.png", dpi=150, bbox_inches="tight"); plt.close()

# Distribusi Residuals
residuals = np.array(y_test) - y_pred
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=40, kde=True, color="#27ae60")
plt.xlabel("Residuals"); plt.ylabel("Frekuensi")
plt.title("Distribusi Residuals — Random Forest", fontweight="bold")
plt.tight_layout(); plt.savefig("residual_dist_forest.png", dpi=150, bbox_inches="tight"); plt.close()

# Demand Aktual vs Prediksi per Category
df_eval = df_clean.iloc[X_test.index - X_train.shape[0] if hasattr(X_test,"index") else X_test.index].copy() if hasattr(X_test,"index") else None
try:
    idx_test = X_test.index
    df_result = df_clean.loc[idx_test].copy()
    df_result["Predicted"] = y_pred
    cat_actual = df_result.groupby("Category")["Demand"].mean()
    cat_pred   = df_result.groupby("Category")["Predicted"].mean()
    cat_df = pd.DataFrame({"Actual": cat_actual, "Predicted": cat_pred})
    cat_df.plot(kind="bar", figsize=(10,6), color=["#3498db","#e74c3c"])
    plt.title("Rata-rata Demand Aktual vs Prediksi per Category — Random Forest", fontweight="bold")
    plt.xlabel("Category"); plt.ylabel("Avg Demand"); plt.xticks(rotation=45)
    plt.legend(["Aktual","Prediksi"]); plt.tight_layout()
    plt.savefig("demand_per_category.png", dpi=150, bbox_inches="tight"); plt.close()
    print("✅ demand_per_category.png disimpan")
except Exception as e:
    print(f"⚠️  Gagal membuat demand_per_category.png: {e}")

print("✅ Semua visualisasi disimpan")

# ─── 8. CONFIDENCE INTERVAL 95% ─────────────────────────────────────────────
print("\n📐 Menghitung CI 95% dari individual tree predictions ...")
X_test_transformed = best_pipeline.named_steps["preprocessor"].transform(X_test)
all_preds = np.array([
    tree.predict(X_test_transformed)
    for tree in best_pipeline.named_steps["model"].estimators_
])
mean_pred = all_preds.mean(axis=0)
std_pred  = all_preds.std(axis=0)
ci_lower  = mean_pred - 1.96 * std_pred
ci_upper  = mean_pred + 1.96 * std_pred

print("   Contoh 5 prediksi pertama + CI 95%:")
print(f"  {'No':>3} | {'Actual':>8} | {'Pred':>10} | {'CI Lower':>10} | {'CI Upper':>10}")
for i in range(5):
    print(f"  {i+1:>3} | {list(y_test)[i]:>8.2f} | {mean_pred[i]:>10.2f} | {ci_lower[i]:>10.2f} | {ci_upper[i]:>10.2f}")

# ─── 9. YoY ANALYSIS ────────────────────────────────────────────────────────
print("\n📅 Year-on-Year Analysis — Rata-rata Demand per Tahun (Semua Kategori):")
yoy = df_clean.groupby("Year")["Demand"].mean().reset_index()
yoy.columns = ["Year","Avg Demand"]
yoy["Growth (%)"] = yoy["Avg Demand"].pct_change() * 100
print(yoy.to_string(index=False))

# ─── 10. SAVE MODEL ─────────────────────────────────────────────────────────
joblib.dump(best_pipeline, "forest_model.pkl")
print("\n✅ Model Random Forest berhasil disimpan ke forest_model.pkl")
print("🎉 FOREST_mod.py selesai!")
