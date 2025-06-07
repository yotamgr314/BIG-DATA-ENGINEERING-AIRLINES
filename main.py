import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ——————————————
# 1) Load all layers
# ——————————————
df_gold = pd.read_csv("GOLD/gold_monthly_performance.csv")

silver_tix = pd.read_csv(
    "SILVER/silver_tickets.csv",
    parse_dates=["booking_date"]
)

bronze_booked = pd.read_csv("BRONZE/booked_ticket.csv")

# Merge to get ticket_class + price + flight + booking_date
tix = (
    silver_tix
    .merge(
        bronze_booked[["ticket_id","ticket_class"]],
        on="ticket_id",
        how="left"
    )
    .assign(
        year=lambda df: df["booking_date"].dt.year,
        month=lambda df: df["booking_date"].dt.month
    )
)

# ——————————————
# 2) Streamlit UI
# ——————————————
st.title("📊 Flight Analytics Dashboard (Gold Layer)")

# ——————————————————————————————
# A) Year filter for ***gold*** dashboard
# ——————————————————————————————
year_gold = st.selectbox("Select Year (Gold Data)", sorted(df_gold["year"].unique()))
year_gold_df = df_gold[df_gold["year"] == year_gold]

# … existing gold plots …
st.subheader("✈️ Total Flights per Month")
fig1, ax1 = plt.subplots()
ax1.plot(year_gold_df["month"], year_gold_df["total_flights"], marker="o")
ax1.set_xlabel("Month")
ax1.set_ylabel("Total Flights")
st.pyplot(fig1)

st.subheader("⏱ Average Delay per Month")
fig2, ax2 = plt.subplots()
ax2.plot(year_gold_df["month"], year_gold_df["average_delay"], marker="s")
ax2.set_xlabel("Month")
ax2.set_ylabel("Average Delay (min)")
st.pyplot(fig2)

st.subheader("💰 Total Profit per Month")
fig3, ax3 = plt.subplots()
ax3.bar(year_gold_df["month"], year_gold_df["total_profit"])
ax3.set_xlabel("Month")
ax3.set_ylabel("Total Profit")
st.pyplot(fig3)

st.subheader("🌍 Most Popular Destinations")
st.bar_chart(df_gold["most_popular_destination"].value_counts())

st.subheader("📱 Most Popular Order Methods")
st.bar_chart(df_gold["most_popular_order_method"].value_counts())

# ——————————————
# B) Which ticket class is most profitable each month?
# ——————————————
st.subheader("💳 Most Profitable Ticket Class per Month")

years_tix = sorted(tix["year"].unique())
if years_tix:
    year_tix = st.selectbox("Select Year (Ticket Revenue)", years_tix, key="ticket_year")
    df_tix_year = tix[tix["year"] == year_tix]

    rev_by_class = (
        df_tix_year
        .groupby(["month","ticket_class"])["ticket_price"]
        .sum()
        .reset_index()
    )

    # pivot into months×classes, ensure months 1–12 AND the three correct classes
    pivot = (
        rev_by_class
        .pivot(index="month", columns="ticket_class", values="ticket_price")
        .fillna(0)
        .reindex(index=range(1,13), fill_value=0)
        .reindex(columns=["Economy","Business","Premium"], fill_value=0)
    )

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

# ——————————————
# C) Per‐Flight Seat Breakdown
# ——————————————
st.subheader("🪑 Seat Breakdown per Flight")

all_flights = sorted(tix["flight_id"].unique())
if all_flights:
    flight_choice = st.selectbox("Select Flight", all_flights, key="flight_id")
    fx = tix[tix["flight_id"] == flight_choice]

    # only these three classes exist now
    all_classes = ["Economy","Business","Premium"]
    counts     = fx["ticket_class"].value_counts().reindex(all_classes, fill_value=0)

    st.write(f"**Flight ID:** {flight_choice}")
    st.bar_chart(counts)
else:
    st.warning("No flight tickets to show.")

# ——————————————————————————————
# C) Per-Flight Seat Breakdown
# ——————————————————————————————
st.subheader("🪑 Seat Breakdown per Flight")

all_flights = sorted(tix["flight_id"].unique())
if not all_flights:
    st.warning("No flight tickets to show.")
else:
    flight_choice = st.selectbox("Select Flight", all_flights, key="flight_id")
    fx = tix[tix["flight_id"] == flight_choice]

    # ensure all classes are present
    all_classes = ["First", "Business", "Economy"]
    counts = fx["ticket_class"].value_counts().reindex(all_classes, fill_value=0)

    st.write(f"**Flight ID:** {flight_choice}")
    st.bar_chart(counts)
