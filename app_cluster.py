# app_cluster.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# تنظیم صفحه
st.set_page_config(page_title="پیش‌بینی خوشه مشتری", layout="centered")

# عنوان
st.title("🔮 پیش‌بینی خوشه مشتری")
st.markdown("---")

# بارگذاری مدل‌ها
@st.cache_resource
def load_models():
    rf = joblib.load('cluster_rf_model.pkl')
    scaler = joblib.load('cluster_scaler.pkl')
    return rf, scaler

rf, scaler = load_models()

# اطلاعات خوشه‌ها
cluster_info = {
    0: {"name": "🟢 مشتریان طلایی", "strategy": "تخفیف ویژه 15% + خدمات VIP"},
    1: {"name": "🟡 مشتریان متوسط", "strategy": "ایمیل مناسبتی + برنامه پاداش"},
    2: {"name": "🔴 مشتریان خفته", "strategy": "کد تخفیف 25% برای بازگشت"}
}

# فرم ورودی
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        recency = st.number_input("📅 آخرین خرید (چند روز پیش)", min_value=0, value=30)
        frequency = st.number_input("📊 تعداد کل سفارش‌ها", min_value=1, value=25)
    
    with col2:
        monetary = st.number_input("💰 مجموع فروش (دلار)", min_value=0, value=15000)
    
    submitted = st.form_submit_button("🔍 پیش‌بینی خوشه")

# پیش‌بینی
if submitted:
    features = np.array([[recency, frequency, monetary]])
    features_scaled = scaler.transform(features)
    cluster = rf.predict(features_scaled)[0]
    info = cluster_info[cluster]
    
    st.markdown("---")
    st.success(f"### {info['name']}")
    st.info(f"💡 **استراتژی پیشنهادی:** {info['strategy']}")

# راهنما
with st.expander("📖 راهنمای مقادیر"):
    st.markdown("""
    | خوشه | Recency (روز) | Frequency (تعداد) | Monetary (فروش) |
    |------|--------------|-------------------|-----------------|
    | طلایی | کمتر از 30 | بیشتر از 50 | بیشتر از 15000 |
    | متوسط | 30 تا 100 | 20 تا 50 | 3000 تا 15000 |
    | خفته | بیشتر از 100 | کمتر از 20 | کمتر از 3000 |
    """)