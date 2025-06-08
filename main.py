import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ——————————————
# 1) Load only GOLD-layer data
# ——————————————
df_gold = pd.read_csv("GOLD/gold_monthly_performance.csv")

# ——————————————
# 2) Streamlit UI
# ——————————————
st.title("📊 Flight Analytics Dashboard (Gold Layer)")

# ——————————————————————————————
# A) Gold-layer Year filter & core KPIs
# ——————————————————————————————
year_gold = st.selectbox(
    "Select Year (Gold Data)",
    sorted(df_gold["year"].unique())
)
df_year_gold = df_gold[df_gold["year"] == year_gold]

st.subheader("✈️ Total Flights per Month")
fig1, ax1 = plt.subplots()
ax1.plot(df_year_gold["month"], df_year_gold["total_flights"], marker="o")
ax1.set_xlabel("Month"); ax1.set_ylabel("Total Flights")
st.pyplot(fig1)

st.subheader("⏱ Average Delay per Month")
fig2, ax2 = plt.subplots()
ax2.plot(df_year_gold["month"], df_year_gold["average_delay"], marker="s", color="orange")
ax2.set_xlabel("Month"); ax2.set_ylabel("Average Delay (minutes)")
st.pyplot(fig2)

st.subheader("💰 Total Profit per Month")
fig3, ax3 = plt.subplots()
ax3.bar(df_year_gold["month"], df_year_gold["total_profit"], color="green")
ax3.set_xlabel("Month"); ax3.set_ylabel("Total Profit")
st.pyplot(fig3)

# ——————————————————————————————
# B) Gold-layer Dimension Analyses
# ——————————————————————————————
st.subheader("🌍 Most Popular Destinations")
st.bar_chart(df_gold["most_popular_destination"].value_counts())

st.subheader("📱 Most Popular Order Methods")
st.bar_chart(df_gold["most_popular_order_method"].value_counts())
