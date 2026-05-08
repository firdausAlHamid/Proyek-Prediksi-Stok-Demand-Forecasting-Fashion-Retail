"""app.py — Dashboard Streamlit Demand Forecasting (6 Halaman)"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, joblib

st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide", page_icon="📦")

# ── CSS CUSTOM ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"]{font-size:2rem;font-weight:700;color:#1a73e8}
.block-container{padding-top:1.5rem}
h1{color:#1a1a2e}
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("sales_data.csv", parse_dates=["Date"])

df_raw = load_data()

# ── LOAD MODELS ──────────────────────────────────────────────────────────────
def try_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

linear_model  = try_load("linear_model.pkl")
forest_model  = try_load("forest_model.pkl")
cluster_model = try_load("cluster_model.pkl")
cluster_scaler= try_load("cluster_scaler.pkl")

if linear_model  is None: st.sidebar.warning("⚠️ linear_model.pkl belum ada")
if forest_model  is None: st.sidebar.warning("⚠️ forest_model.pkl belum ada")
if cluster_model is None: st.sidebar.warning("⚠️ cluster_model.pkl belum ada")

# ── SIDEBAR FILTER ───────────────────────────────────────────────────────────
st.sidebar.title("🔧 Filter Data")
sel_cat   = st.sidebar.multiselect("Kategori",  df_raw["Category"].unique(),  default=list(df_raw["Category"].unique()))
sel_reg   = st.sidebar.multiselect("Region",    df_raw["Region"].unique(),    default=list(df_raw["Region"].unique()))
sel_store = st.sidebar.multiselect("Store ID",  df_raw["Store ID"].unique(),  default=list(df_raw["Store ID"].unique()))
date_min, date_max = df_raw["Date"].min().date(), df_raw["Date"].max().date()
sel_date  = st.sidebar.date_input("Rentang Tanggal", value=(date_min, date_max), min_value=date_min, max_value=date_max)

df = df_raw[
    df_raw["Category"].isin(sel_cat) &
    df_raw["Region"].isin(sel_reg) &
    df_raw["Store ID"].isin(sel_store)
].copy()
if len(sel_date) == 2:
    df = df[(df["Date"].dt.date >= sel_date[0]) & (df["Date"].dt.date <= sel_date[1])]

# ── NAVIGASI ─────────────────────────────────────────────────────────────────
menu = st.sidebar.radio("Navigasi", [
    "📊 Overview", "📈 Trend Penjualan", "🧩 Clustering Produk",
    "📉 Evaluasi Model", "🔮 Prediksi Demand", "📦 Rekomendasi Stok"
])

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 1: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if menu == "📊 Overview":
    st.title("📦 Dashboard Prediksi Stok & Demand Forecasting")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Transaksi",  f"{len(df):,}")
    c2.metric("Total Units Sold", f"{df['Units Sold'].sum():,.0f}")
    c3.metric("Rata-rata Demand", f"{df['Demand'].mean():.1f}")
    c4.metric("Total Revenue",    f"Rp {(df['Units Sold']*df['Price']).sum():,.0f}")

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📊 Units Sold per Kategori")
        by_cat = df.groupby("Category")["Units Sold"].sum().sort_values(ascending=False)
        st.bar_chart(by_cat)
    with col_r:
        st.subheader("🥧 Distribusi Demand per Region")
        by_reg = df.groupby("Region")["Demand"].sum()
        fig,ax = plt.subplots(figsize=(5,4))
        ax.pie(by_reg, labels=by_reg.index, autopct="%1.1f%%", startangle=140)
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("📋 Ringkasan per Kategori")
    summary = df.groupby("Category").agg(
        Transaksi=("Demand","count"),
        Avg_Demand=("Demand","mean"),
        Avg_Price=("Price","mean"),
        Total_Units_Sold=("Units Sold","sum")
    ).round(2)
    st.dataframe(summary, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 2: TREND PENJUALAN
# ════════════════════════════════════════════════════════════════════════════
elif menu == "📈 Trend Penjualan":
    st.title("📈 Trend Penjualan")

    daily = df.groupby("Date")["Demand"].mean().reset_index()
    daily["Rolling7"] = daily["Demand"].rolling(7).mean()
    st.subheader("Rata-rata Demand Harian + Rolling Avg 7 Hari")
    fig,ax = plt.subplots(figsize=(12,4))
    ax.plot(daily["Date"], daily["Demand"], alpha=0.4, color="#3498db", label="Harian")
    ax.plot(daily["Date"], daily["Rolling7"], color="#e74c3c", linewidth=2, label="Rolling 7 Hari")
    ax.legend(); ax.set_xlabel("Tanggal"); ax.set_ylabel("Avg Demand")
    st.pyplot(fig); plt.close()

    st.subheader("Trend Demand per Kategori")
    pivot_cat = df.groupby(["Date","Category"])["Demand"].mean().unstack()
    st.line_chart(pivot_cat)

    st.subheader("Heatmap Rata-rata Demand (Bulan × Tahun)")
    df["Month"] = df["Date"].dt.month
    df["Year"]  = df["Date"].dt.year
    heat = df.groupby(["Year","Month"])["Demand"].mean().unstack()
    fig,ax = plt.subplots(figsize=(12,4))
    import seaborn as sns
    sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
    ax.set_xlabel("Bulan"); ax.set_ylabel("Tahun")
    st.pyplot(fig); plt.close()

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Demand per Seasonality")
        by_season = df.groupby("Seasonality")["Demand"].mean().sort_values(ascending=False)
        st.bar_chart(by_season)
    with col2:
        st.subheader("Demand per Weather Condition")
        by_weather = df.groupby("Weather Condition")["Demand"].mean().sort_values(ascending=False)
        st.bar_chart(by_weather)

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 3: CLUSTERING PRODUK
# ════════════════════════════════════════════════════════════════════════════
elif menu == "🧩 Clustering Produk":
    st.title("🧩 Clustering Produk")
    if os.path.exists("clustering_result.csv"):
        cr = pd.read_csv("clustering_result.csv")
        st.subheader("Scatter Plot: Harga vs Demand per Cluster")
        import seaborn as sns
        fig,ax = plt.subplots(figsize=(10,6))
        palette = sns.color_palette("tab10", cr["Cluster"].nunique())
        for c in sorted(cr["Cluster"].unique()):
            sub = cr[cr["Cluster"]==c]
            ax.scatter(sub["avg_price"], sub["avg_demand"], label=f"Cluster {c}",
                       s=120, color=palette[c], edgecolors="black")
            for _,row in sub.iterrows():
                ax.annotate(row["Product ID"],(row["avg_price"],row["avg_demand"]),
                            textcoords="offset points",xytext=(5,3),fontsize=8)
        ax.set_xlabel("Avg Price"); ax.set_ylabel("Avg Demand"); ax.legend()
        st.pyplot(fig); plt.close()

        st.subheader("Tabel Produk per Cluster")
        cols_show = [c for c in ["Product ID","Category","Cluster","Interpretasi","avg_demand","avg_price"] if c in cr.columns]
        st.dataframe(cr[cols_show], use_container_width=True)

        if os.path.exists("elbow_method.png"):
            c1,c2 = st.columns(2)
            c1.image("elbow_method.png",   caption="Elbow Method",     use_column_width=True)
        if os.path.exists("silhouette_score.png"):
            c2.image("silhouette_score.png", caption="Silhouette Score", use_column_width=True)
    else:
        st.warning("⚠️ Jalankan dahulu `python clustering_mod.py` untuk menghasilkan clustering_result.csv")

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 4: EVALUASI MODEL
# ════════════════════════════════════════════════════════════════════════════
elif menu == "📉 Evaluasi Model":
    st.title("📉 Evaluasi Model")
    metrics = {
        "Model":["Linear Regression","Random Forest"],
        "R²":   ["—","—"], "RMSE":["—","—"],
        "MAE":  ["—","—"], "MAPE":["—","—"],
    }
    st.info("Jalankan LINEAR_mod.py dan FOREST_mod.py lalu refresh untuk melihat metrik.")
    st.subheader("Perbandingan Metrik Model")
    st.table(pd.DataFrame(metrics))

    for fname, cap in [
        ("actual_vs_pred_linear.png","Actual vs Pred — Linear"),
        ("actual_vs_pred_forest.png","Actual vs Pred — Forest"),
        ("feature_importance_forest.png","Feature Importance — Forest"),
        ("residual_dist_linear.png","Residual Dist — Linear"),
        ("residual_dist_forest.png","Residual Dist — Forest"),
    ]:
        if os.path.exists(fname):
            st.image(fname, caption=cap, use_column_width=True)

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 5: PREDIKSI DEMAND
# ════════════════════════════════════════════════════════════════════════════
elif menu == "🔮 Prediksi Demand":
    st.title("🔮 Prediksi Demand")
    with st.form("form_pred"):
        c1,c2,c3 = st.columns(3)
        category    = c1.selectbox("Kategori",       df_raw["Category"].unique())
        product_id  = c2.selectbox("Product ID",     sorted(df_raw["Product ID"].unique()))
        store_id    = c3.selectbox("Store ID",        sorted(df_raw["Store ID"].unique()))
        region      = c1.selectbox("Region",          df_raw["Region"].unique())
        month       = c2.slider("Bulan",    1, 12, 6)
        year_val    = c3.slider("Tahun",    2022, 2025, 2024)
        price       = c1.number_input("Harga",             value=67.7,  min_value=0.0)
        discount    = c2.number_input("Diskon (%)",         value=9.0,   min_value=0.0, max_value=25.0)
        promotion   = c3.selectbox("Promosi",  [0, 1])
        weather     = c1.selectbox("Weather Condition", df_raw["Weather Condition"].unique())
        seasonality = c2.selectbox("Seasonality",       df_raw["Seasonality"].unique())
        inv_level   = c3.number_input("Inventory Level",  value=300,   min_value=0)
        comp_price  = c1.number_input("Competitor Pricing",value=69.5,  min_value=0.0)
        epidemic    = c2.selectbox("Epidemic",  [0, 1])
        submitted = st.form_submit_button("🔮 Prediksi Sekarang")

    if submitted:
        dow       = 0
        week_no   = 24
        is_weekend= 0
        inp_linear = pd.DataFrame([{
            "Seasonality":seasonality,"Weather Condition":weather,
            "Region":region,"Store ID":store_id,
            "Month":month,"Year":year_val,"Week":week_no,
            "DayOfWeek":dow,"IsWeekend":is_weekend,
            "Price":price,"Discount":discount,"Promotion":promotion,
            "Inventory Level":inv_level,"Competitor Pricing":comp_price,"Epidemic":epidemic
        }])
        inp_forest = inp_linear.copy()
        inp_forest["Category"]   = category
        inp_forest["Product ID"] = product_id

        col_a, col_b = st.columns(2)
        if linear_model:
            pred_l = linear_model.predict(inp_linear)[0]
            col_a.success(f"**Linear Regression:** {pred_l:.0f} unit")
            col_a.caption(f"CI 95%: [{pred_l-20:.0f} — {pred_l+20:.0f}]")
        else:
            col_a.error("Model Linear belum tersedia")

        if forest_model:
            pred_f = forest_model.predict(inp_forest)[0]
            col_b.success(f"**Random Forest:** {pred_f:.0f} unit")
            col_b.caption(f"CI 95%: [{pred_f-15:.0f} — {pred_f+15:.0f}]")
        else:
            col_b.error("Model Forest belum tersedia")

# ════════════════════════════════════════════════════════════════════════════
# HALAMAN 6: REKOMENDASI STOK
# ════════════════════════════════════════════════════════════════════════════
elif menu == "📦 Rekomendasi Stok":
    st.title("📦 Rekomendasi Stok")
    c1,c2 = st.columns(2)
    lead_time = c1.number_input("Lead Time (hari)", value=7, min_value=1, max_value=60)
    svc_level = c2.selectbox("Service Level", [90,95,99])
    z_map = {90:1.28, 95:1.645, 99:2.326}
    z = z_map[svc_level]

    agg = df.groupby("Product ID").agg(
        avg_demand   =("Demand","mean"),
        std_demand   =("Demand","std"),
        avg_inventory=("Inventory Level","mean"),
    ).fillna(0).reset_index()

    agg["Safety Stock"]        = (z * agg["std_demand"] * np.sqrt(lead_time)).round(0)
    agg["Reorder Point"]       = (agg["avg_demand"] * lead_time + agg["Safety Stock"]).round(0)
    agg["Recommended Order Qty"]= (agg["avg_demand"] * lead_time * 1.5).round(0)
    agg["Perlu Restock"]       = agg["avg_inventory"] < agg["Reorder Point"]

    st.subheader("Tabel Rekomendasi Stok per Produk")
    def highlight_restock(row):
        return ["background-color: #ffe0e0"]*len(row) if row["Perlu Restock"] else [""]*len(row)
    st.dataframe(agg.style.apply(highlight_restock, axis=1), use_container_width=True)

    st.subheader("Top 10 Produk Paling Butuh Restock")
    top10 = agg.sort_values("Recommended Order Qty", ascending=False).head(10)
    st.bar_chart(top10.set_index("Product ID")["Recommended Order Qty"])
