import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ——————————————
# 1) Load data
# ——————————————
df_gold = pd.read_csv("GOLD/gold_monthly_performance.csv")

# silver_tickets.csv now already has ticket_class
silver_tix = pd.read_csv(
    "SILVER/silver_tickets.csv",
    parse_dates=["booking_date"]
)

# Build tix DataFrame straight from silver_tix
tix = silver_tix.copy()
tix["year"]  = tix["booking_date"].dt.year
tix["month"] = tix["booking_date"].dt.month

# ——————————————
# 2) Streamlit UI
# ——————————————
st.title("📊 Flight Analytics Dashboard (Gold Layer)")

# ——————————————————————————————
# A) Gold-layer Year filter & plots
# ——————————————————————————————
year_gold = st.selectbox(
    "Select Year (Gold Data)",
    sorted(df_gold["year"].unique()),
    key="gold_year"
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

st.subheader("🌍 Most Popular Destinations")
st.bar_chart(df_gold["most_popular_destination"].value_counts())

st.subheader("📱 Most Popular Order Methods")
st.bar_chart(df_gold["most_popular_order_method"].value_counts())

# ——————————————————————————————
# B) Most profitable ticket class per month
# ——————————————————————————————
st.subheader("💳 Most Profitable Ticket Class per Month")

years_tix = sorted(tix["year"].unique())
if years_tix:
    year_tix = st.selectbox(
        "Select Year (Ticket Revenue)",
        years_tix,
        key="ticket_year"
    )
    df_tix_year = tix[tix["year"] == year_tix]

    # Sum revenue by month & class
    rev_by_class = (
        df_tix_year
        .groupby(["month","ticket_class"])["ticket_price"]
        .sum()
        .reset_index()
    )

    # Pivot so months 1–12 appear and only the three real classes
    pivot = (
        rev_by_class
        .pivot(index="month", columns="ticket_class", values="ticket_price")
        .fillna(0)
        .reindex(index=range(1,13), fill_value=0)
        .reindex(columns=["Economy","Business","Premium"], fill_value=0)
    )

    # Determine best class (blank if no revenue)
    total_rev  = pivot.max(axis=1)
    best_class = pivot.idxmax(axis=1).where(total_rev > 0, "")

    best = pd.DataFrame({
        "month": pivot.index,
        "best_class": best_class.values,
        "total_revenue": total_rev.values
    }).set_index("month")

    st.table(best)
    st.bar_chart(pivot)
else:
    st.warning("No ticket data available.")

# ——————————————————————————————
# C) Seat breakdown per flight
# ——————————————————————————————
st.subheader("🪑 Seat Breakdown per Flight")

all_flights = sorted(tix["flight_id"].unique())
if all_flights:
    flight_choice = st.selectbox("Select Flight", all_flights, key="flight_id")
    fx = tix[tix["flight_id"] == flight_choice]

    counts = (
        fx["ticket_class"]
        .value_counts()
        .reindex(["Economy","Business","Premium"], fill_value=0)
    )

    st.write(f"**Flight ID:** {flight_choice}")
    st.bar_chart(counts)
else:
    st.warning("No flight tickets to show.")
