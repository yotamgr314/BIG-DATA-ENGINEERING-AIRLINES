import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ——————————————
# 1) Load data
# ——————————————
df_gold = pd.read_csv("GOLD/gold_monthly_performance.csv")

silver_tix = pd.read_csv(
    "SILVER/silver_tickets.csv",
    parse_dates=["booking_date"]
)

df_silver_flights = pd.read_csv(
    "SILVER/silver_flights.csv",
    parse_dates=[
        "estimated_departure_time",
        "estimated_arrival_time",
        "actual_departure_time",
        "actual_arrival_time",
        "report_date"
    ]
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
year_gold = st.selectbox("Select Year (Gold Data)", sorted(df_gold["year"].unique()))
df_year_gold = df_gold[df_gold["year"] == year_gold]

st.subheader("✈️ Total Flights per Month")
fig1, ax1 = plt.subplots()
ax1.plot(df_year_gold["month"], df_year_gold["total_flights"], marker="o")
ax1.set_xlabel("Month")
ax1.set_ylabel("Total Flights")
st.pyplot(fig1)

st.subheader("⏱ Average Delay per Month")
fig2, ax2 = plt.subplots()
ax2.plot(df_year_gold["month"], df_year_gold["average_delay"], marker="s", color="orange")
ax2.set_xlabel("Month")
ax2.set_ylabel("Average Delay (minutes)")
st.pyplot(fig2)

st.subheader("💰 Total Profit per Month")
fig3, ax3 = plt.subplots()
ax3.bar(df_year_gold["month"], df_year_gold["total_profit"], color="green")
ax3.set_xlabel("Month")
ax3.set_ylabel("Total Profit")
st.pyplot(fig3)

st.subheader("🌍 Most Popular Destinations")
st.bar_chart(df_gold["most_popular_destination"].value_counts())

st.subheader("📱 Most Popular Order Methods")
st.bar_chart(df_gold["most_popular_order_method"].value_counts())

# ——————————————————————————————
# B) Most Profitable Ticket Class per Month
# ——————————————————————————————
st.subheader("💳 Most Profitable Ticket Class per Month")
years_tix = sorted(tix["year"].unique())
if years_tix:
    year_tix = st.selectbox("Select Year (Ticket Revenue)", years_tix)
    df_tix_year = tix[tix["year"] == year_tix]
    rev_by_class = (
        df_tix_year
        .groupby(["month","ticket_class"])["ticket_price"]
        .sum()
        .reset_index()
    )
    pivot_rc = (
        rev_by_class
        .pivot(index="month", columns="ticket_class", values="ticket_price")
        .fillna(0)
        .reindex(index=range(1,13), fill_value=0)
        .reindex(columns=["Economy","Business","Premium"], fill_value=0)
    )
    total_rev  = pivot_rc.max(axis=1)
    best_class = pivot_rc.idxmax(axis=1).where(total_rev > 0, "")
    best = pd.DataFrame({
        "month": pivot_rc.index,
        "best_class": best_class.values,
        "total_revenue": total_rev.values
    }).set_index("month")
    st.table(best)
    st.bar_chart(pivot_rc)
else:
    st.warning("No ticket data available.")

# ——————————————————————————————
# C) Seat Breakdown per Flight (as fraction)
# ——————————————————————————————
st.subheader("🪑 Seat Breakdown per Flight")

all_flights = sorted(tix["flight_id"].unique())
if all_flights:
    flight_choice = st.selectbox("Select Flight", all_flights)
    fx = tix[tix["flight_id"] == flight_choice]

    # raw counts
    counts = fx["ticket_class"] \
        .value_counts() \
        .reindex(["Economy","Business","Premium"], fill_value=0)

    # convert to fraction of total
    frac = counts / counts.sum()

    st.write(f"**Flight ID:** {flight_choice}")
    # plot with Y axis 0.0–1.0
    fig, ax = plt.subplots()
    ax.bar(frac.index, frac.values)
    ax.set_ylabel("Fraction of Seats")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
else:
    st.warning("No flight tickets to show.")

# ——————————————————————————————
# D) Average Delays by Route
# ——————————————————————————————
st.subheader("🛣️ Average Delays by Route (Top 10)")
route_delays = (
    df_silver_flights
    .groupby(["airport_origin","airport_destination"])["delay_in_minutes"]
    .mean()
    .reset_index()
    .sort_values("delay_in_minutes", ascending=False)
    .head(10)
)
st.table(route_delays)

# ——————————————————————————————
# F) Route Profitability
# ——————————————————————————————
st.subheader("💼 Route Profitability (Average Profit per Flight, Top 10)")
profit_by_route = (
    df_silver_flights
    .groupby(["airport_origin","airport_destination"])["total_flight_profit"]
    .mean()
    .reset_index()
    .sort_values("total_flight_profit", ascending=False)
    .head(10)
)
st.table(profit_by_route)

# ——————————————————————————————
# G) Passenger Mix by Origin Airport
# ——————————————————————————————
st.subheader("👥 Passenger Mix by Origin Airport")
mix = (
    tix.merge(df_silver_flights[["flight_id","airport_origin"]], on="flight_id")
       .groupby(["airport_origin","ticket_class"])
       .size()
       .unstack(fill_value=0)
)
st.bar_chart(mix)

# ——————————————————————————————
# I) Flight Status by Month (Proportion)
# ——————————————————————————————
st.subheader("🚦 Flight Status by Month (as fraction of total)")

# 1. בוחרים שנה להצגה
status_years = sorted(df_silver_flights["report_date"].dt.year.unique())
year_status = st.selectbox("Select Year (Flight Status)", status_years)

# 2. מסננים לטיסות בשנה שבחרנו ומוסיפים עמודת month
status_df = df_silver_flights[
    df_silver_flights["report_date"].dt.year == year_status
].copy()
status_df["month"] = status_df["report_date"].dt.month

# 3. סופרים כל סטטוס בחודש, מוודאים שכל 12 החודשים יופיעו
status_counts = (
    status_df
    .groupby(["month","status"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=range(1,13), fill_value=0)
)

# 4. מחשבים את השבר (0–1) מכלל הטיסות בחודש
status_frac = status_counts.div(status_counts.sum(axis=1), axis=0)

# 5. מציגים בעמודות ומגבילים ציר Y ל־[0, 1]
fig, ax = plt.subplots()
status_frac.plot(kind="bar", ax=ax)
ax.set_xlabel("Month")
ax.set_ylabel("Fraction of Flights")
ax.set_ylim(0, 1)
ax.legend(title="Status", bbox_to_anchor=(1,1))
st.pyplot(fig)
