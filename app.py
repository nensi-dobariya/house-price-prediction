import streamlit as st
import pickle
import pandas as pd

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
/* Global Styles */
html, body {
    font-size: 18px;
    background-color: #f4f6f9;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Card Style */
.card {
    background-color: #ffffff;
    padding: 28px;
    border-radius: 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.12);
    margin-bottom: 25px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 12px 30px rgba(0,0,0,0.2);
}

/* Titles */
.title {
    text-align: center;
    color: #2E86C1;
    font-size: 50px;
    font-weight: 900;
}

.subtitle {
    text-align: center;
    color: #7f8c8d;
    font-size: 26px;
    margin-bottom: 20px;
}

/* Sidebar Input Labels */
.stSidebar label {
    font-size: 18px !important;
    font-weight: 600;
}

/* Value Display */
.value {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 10px;
}

/* Price Box */
.price-box {
    background: linear-gradient(135deg, #2E86C1, #5DADE2);
    padding: 40px;
    border-radius: 25px;
    color: white;
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    margin-top: 25px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
    transition: transform 0.3s ease;
}
.price-box:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
with open("final_house_model.pkl", "rb") as f:
    model, scaler, features = pickle.load(f)

# ---------------- Title ----------------
st.markdown("""
<h1 class='title'>🏠 House Price Prediction</h1>
<p class='subtitle'>Predict house price using Machine Learning</p>
<hr>
""", unsafe_allow_html=True)

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Enter House Details")

overall_qual = st.sidebar.number_input("Overall Quality (1–10)", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Living Area (sqft)", 300, 6000, 1500)
bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.number_input("Bathrooms", 1, 5, 2)
basement = st.sidebar.number_input("Basement Area (sqft)", 0, 4000, 800)

# ---------------- Preview Card ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📋 Entered House Details")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"<p class='value'>🛏 Bedrooms: {bedrooms}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='value'>🛁 Bathrooms: {bathrooms}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='value'>⭐ Overall Quality: {overall_qual}</p>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<p class='value'>📐 Living Area: {gr_liv_area} sqft</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='value'>🏠 Basement Area: {basement} sqft</p>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Prediction ----------------
if st.button("💰 Predict House Price"):

    input_df = pd.DataFrame([[  
        overall_qual,
        gr_liv_area,
        bedrooms,
        bathrooms,
        basement
    ]], columns=features)

    input_scaled = scaler.transform(input_df)
    price = model.predict(input_scaled)

    st.markdown(f"""
    <div class='price-box'>
        🏷 Estimated House Price <br><br>
        ₹ {round(price[0], 2)}
    </div>
    """, unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("""
<hr>
<p style='text-align:center; color:gray; font-size:18px;'>
Developed using Streamlit & Machine Learning
</p>
""", unsafe_allow_html=True)
