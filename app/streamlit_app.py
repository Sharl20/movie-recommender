
import streamlit as st
import requests

st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommendation System")

API_URL = "https://your-app.onrender.com"  # هنغيره بعد الـ deploy

st.sidebar.header("Settings")
user_id   = st.sidebar.number_input("User ID", min_value=1, max_value=943, value=1)
n         = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
cf_weight = st.sidebar.slider("CF Weight", 0.0, 1.0, 0.7)
cb_weight = round(1 - cf_weight, 1)
st.sidebar.write(f"CB Weight: {cb_weight}")

tab1, tab2 = st.tabs(["🤝 Hybrid", "👥 Collaborative Filtering"])

with tab1:
    if st.button("Get Hybrid Recommendations"):
        with st.spinner("Getting recommendations..."):
            res = requests.post(f"{API_URL}/recommend/hybrid", json={
                "user_id": user_id, "n": n,
                "cf_weight": cf_weight, "cb_weight": cb_weight
            })
            if res.status_code == 200:
                recs = res.json()["recommendations"]
                for i, r in enumerate(recs, 1):
                    st.write(f"**{i}.** {r['title']} — Score: {r['score']:.3f}")
            else:
                st.error("API Error!")

with tab2:
    if st.button("Get CF Recommendations"):
        with st.spinner("Getting recommendations..."):
            res = requests.post(f"{API_URL}/recommend/cf", json={
                "user_id": user_id, "n": n
            })
            if res.status_code == 200:
                recs = res.json()["recommendations"]
                for i, r in enumerate(recs, 1):
                    st.write(f"**{i}.** {r['title']}")
            else:
                st.error("API Error!")
