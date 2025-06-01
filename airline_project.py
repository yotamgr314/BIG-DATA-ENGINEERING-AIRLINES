import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------
# Create directories if they don't exist:
#   - bronze/      (raw ingested data)
#   - silver/      (cleaned & enriched data)
#   - gold/        (aggregated KPIs & ML features)
# ---------------------------------------------------
for folder in ["bronze", "silver", "gold"]:
    os.makedirs(folder, exist_ok=True)

#########################################
# BRONZE LAYER - RAW DATA INGESTION (Dirty Data)
#########################################
# In the Bronze layer, raw data is ingested exactly as received.
# This includes missing or invalid values (e.g., missing departure times, non-numeric delays).

# 1. Flight Data (with Dirty Data) + Gate
num_flights = 1000  # Adjust as needed
airports = ["JFK", "LAX", "ORD", "ATL", "DFW"]

# Define a set of possible gates for each airport
gates_by_airport = {
    "JFK": [f"JFK_G{n}" for n in range(1, 11)],
    "LAX": [f"LAX_G{n}" for n in range(1, 11)],
    "ORD": [f"ORD_G{n}" for n in range(1, 11)],
    "ATL": [f"ATL_G{n}" for n in range(1, 11)],
    "DFW": [f"DFW_G{n}" for n in range(1, 11)]
}

flight_ids = [f"F{str(i).zfill(4)}" for i in range(1, num_flights + 1)]
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
total_seconds = int((end_date - start_date).total_seconds())

departure_times = []
arrival_times = []
delay_minutes = []
airport_choices = []
gate_choices = []

for _ in range(num_flights):
    # Generate a normal departure and arrival time
    dep = start_date + timedelta(seconds=random.randint(0, total_seconds))
    dur = timedelta(hours=random.randint(1, 5))
    arr = dep + dur

    # Introduce dirty data:
    # 5% chance: missing departure time (empty string)
    dep_str = "" if random.random() < 0.05 else dep.strftime("%Y-%m-%d %H:%M:%S")

    # 5% chance: non-numeric delay_minutes (set to "error")
    delay = "error" if random.random() < 0.05 else random.randint(0, 60)

    airport_choice = random.choice(airports)
    gate_choice = random.choice(gates_by_airport[airport_choice])

    departure_times.append(dep_str)
    arrival_times.append(arr.strftime("%Y-%m-%d %H:%M:%S"))
    delay_minutes.append(delay)
    airport_choices.append(airport_choice)
    gate_choices.append(gate_choice)

df_flights = pd.DataFrame({
    "flight_id": flight_ids,
    "departure_time": departure_times,
    "arrival_time": arrival_times,
    "delay_minutes": delay_minutes,
    "airport": airport_choices,
    "gate": gate_choices
})
df_flights.to_csv("bronze/flights.csv", index=False)
print(f"bronze/flights.csv generated with {num_flights} records.")

# 2. Sensor Data (Simulated Streaming with Dirty Data)
sensor_types = [
    {"type": "Engine Temperature", "min": 500, "max": 700},
    {"type": "Fuel Level", "min": 20, "max": 100},
    {"type": "Vibration", "min": 0, "max": 5},
    {"type": "Air Pressure", "min": 90, "max": 110},
    {"type": "Airspeed", "min": 300, "max": 600}
]
sensor_records = []
for fid, dep, arr in zip(df_flights["flight_id"], df_flights["departure_time"], df_flights["arrival_time"]):
    # Try to parse departure and arrival; if invalid, skip sensor generation
    try:
        dep_dt = datetime.strptime(dep, "%Y-%m-%d %H:%M:%S")
        arr_dt = datetime.strptime(arr, "%Y-%m-%d %H:%M:%S")
    except Exception:
        continue
    for sensor in sensor_types:
        num_records = random.randint(1, 3)
        for _ in range(num_records):
            sensor_id = f"{sensor['type'][:2].upper()}{random.randint(1000, 9999)}"
            event_time = dep_dt + timedelta(seconds=random.randint(0, int((arr_dt - dep_dt).total_seconds())))
            # 5% chance: missing sensor_value (None)
            if random.random() < 0.05:
                sensor_value = None
            else:
                sensor_value = round(random.uniform(sensor["min"], sensor["max"]), 2)
            sensor_records.append({
                "sensor_id": sensor_id,
                "flight_id": fid,
                "sensor_type": sensor["type"],
                "sensor_value": sensor_value,
                "event_time": event_time.strftime("%Y-%m-%d %H:%M:%S")
            })
df_sensors = pd.DataFrame(sensor_records)
df_sensors.to_csv("bronze/sensors.csv", index=False)
print(f"bronze/sensors.csv generated with {len(df_sensors)} records.")

# 3. Customer Data (SCD Type 2) - Assumed Clean
num_customers = 500
first_names = ["John", "Jane", "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson"]
loyalty_tiers = ["Silver", "Gold", "Platinum"]
customer_records = []
for i in range(1, num_customers + 1):
    customer_id = f"C{str(i).zfill(4)}"
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    if random.random() < 0.3:
        start_date_c = datetime(2021, 1, 1) + timedelta(days=random.randint(0, 730))
        change_date = start_date_c + timedelta(days=random.randint(30, 365))
        tier1 = random.choice(loyalty_tiers)
        tier2_choices = [t for t in loyalty_tiers if t != tier1]
        tier2 = random.choice(tier2_choices)
        record1 = {
            "customer_id": customer_id,
            "first_name": first_name,
            "last_name": last_name,
            "loyalty_tier": tier1,
            "effective_date": start_date_c.strftime("%Y-%m-%d"),
            "end_date": change_date.strftime("%Y-%m-%d"),
            "current_flag": "N"
        }
        record2 = {
            "customer_id": customer_id,
            "first_name": first_name,
            "last_name": last_name,
            "loyalty_tier": tier2,
            "effective_date": change_date.strftime("%Y-%m-%d"),
            "end_date": None,
            "current_flag": "Y"
        }
        customer_records.append(record1)
        customer_records.append(record2)
    else:
        start_date_c = datetime(2021, 1, 1) + timedelta(days=random.randint(0, 730))
        record = {
            "customer_id": customer_id,
            "first_name": first_name,
            "last_name": last_name,
            "loyalty_tier": random.choice(loyalty_tiers),
            "effective_date": start_date_c.strftime("%Y-%m-%d"),
            "end_date": None,
            "current_flag": "Y"
        }
        customer_records.append(record)
df_customers = pd.DataFrame(customer_records)
df_customers.to_csv("bronze/customers.csv", index=False)
print(f"bronze/customers.csv generated with {len(df_customers)} records.")

# 4. Bookings Data (Marketing Data Mart)
#    This table links customers to flights and includes booking details.
booking_channels = ["Online", "TravelAgency", "MobileApp"]
booking_records = []
for fid in df_flights["flight_id"]:
    # Each flight can have 0-5 bookings randomly
    num_bookings = random.randint(0, 5)
    for _ in range(num_bookings):
        booking_id = f"B{random.randint(100000, 999999)}"
        customer_id = f"C{str(random.randint(1, num_customers)).zfill(4)}"
        # Booking date is some date before departure_time (up to 60 days prior)
        dep_str = df_flights.loc[df_flights["flight_id"] == fid, "departure_time"].values[0]
        try:
            dep_dt = datetime.strptime(dep_str, "%Y-%m-%d %H:%M:%S")
            booking_dt = dep_dt - timedelta(days=random.randint(1, 60))
        except Exception:
            # If departure_time invalid, skip booking
            continue
        ticket_price = round(random.uniform(100, 1000), 2)
        passenger_count = random.randint(1, 4)
        channel = random.choice(booking_channels)
        booking_records.append({
            "booking_id": booking_id,
            "customer_id": customer_id,
            "flight_id": fid,
            "booking_date": booking_dt.strftime("%Y-%m-%d"),
            "ticket_price": ticket_price,
            "passenger_count": passenger_count,
            "booking_channel": channel
        })
df_bookings = pd.DataFrame(booking_records)
df_bookings.to_csv("bronze/bookings.csv", index=False)
print(f"bronze/bookings.csv generated with {len(df_bookings)} records.")

# 5. Weather Events Data (Synthetic for Flight Routes, including wind direction)
weather_conditions = ["Clear", "Sunny", "Cloudy", "Rain", "Storm"]
weather_events = []
wind_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
for idx, row in df_flights.iterrows():
    fid = row["flight_id"]
    dep = row["departure_time"]
    arr = row["arrival_time"]
    # Parse departure/arrival
    try:
        dep_dt = datetime.strptime(dep, "%Y-%m-%d %H:%M:%S")
        arr_dt = datetime.strptime(arr, "%Y-%m-%d %H:%M:%S")
    except Exception:
        continue
    # Generate a few sample points along the route (3-6 points)
    num_points = random.randint(3, 6)
    route_duration = (arr_dt - dep_dt).total_seconds()
    for _ in range(num_points):
        # Random time between departure and arrival
        rand_seconds = random.randint(0, int(route_duration))
        evt_time = dep_dt + timedelta(seconds=rand_seconds)
        # Random coordinates simulated between two airports (for simplicity)
        latitude = round(random.uniform(20.0, 60.0), 4)
        longitude = round(random.uniform(-130.0, -60.0), 4)
        altitude = round(random.uniform(3000, 40000), 2)  # in feet
        temperature = round(random.uniform(-20, 30), 1)
        wind_speed = round(random.uniform(0, 150), 1)
        direction = random.choice(wind_directions)
        condition = random.choice(weather_conditions)
        # 5% chance: missing temperature or wind_speed
        if random.random() < 0.05:
            temperature = None
        if random.random() < 0.05:
            wind_speed = None
        weather_events.append({
            "flight_id": fid,
            "event_time": evt_time.strftime("%Y-%m-%d %H:%M:%S"),
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "temperature": temperature,
            "wind_speed": wind_speed,
            "wind_direction": direction,
            "weather_condition": condition
        })
df_weather_events = pd.DataFrame(weather_events)
df_weather_events.to_csv("bronze/weather_events.csv", index=False)
print(f"bronze/weather_events.csv generated with {len(df_weather_events)} records.")

#########################################
# SILVER LAYER - CLEANED & ENRICHED DATA
#########################################
# In the Silver layer, the raw Bronze data is cleaned, standardized, and enriched.

# Load Bronze CSV files
df_flights        = pd.read_csv("bronze/flights.csv")
df_customers      = pd.read_csv("bronze/customers.csv")
df_bookings       = pd.read_csv("bronze/bookings.csv")
df_sensors        = pd.read_csv("bronze/sensors.csv")
df_weather_events = pd.read_csv("bronze/weather_events.csv")

# Standardize and Convert Data Types in Flights Data
df_flights["departure_time"] = pd.to_datetime(df_flights["departure_time"], errors='coerce')
df_flights["arrival_time"]   = pd.to_datetime(df_flights["arrival_time"], errors='coerce')
# Drop rows with invalid/missing key date/time values
df_flights = df_flights.dropna(subset=["departure_time", "arrival_time"])
# Convert delay_minutes to numeric and drop invalid rows
df_flights["delay_minutes"] = pd.to_numeric(df_flights["delay_minutes"], errors='coerce')
df_flights = df_flights.dropna(subset=["delay_minutes"])

# Derive flight_date from departure_time
df_flights["flight_date"] = df_flights["departure_time"].dt.date

# Calculate flight_duration in minutes
df_flights["flight_duration"] = (
    (df_flights["arrival_time"] - df_flights["departure_time"])
    .dt.total_seconds() / 60
)

# 1. Enrich: Link flights to bookings (Data Mart)
df_bookings["booking_date"] = pd.to_datetime(df_bookings["booking_date"], errors='coerce').dt.date
df_customers["effective_date"] = pd.to_datetime(df_customers["effective_date"], errors='coerce').dt.date
df_customers["end_date"]       = pd.to_datetime(df_customers["end_date"], errors='coerce').dt.date

df_bookings = pd.merge(df_bookings, df_customers, on="customer_id", how="left")
df_bookings = df_bookings[
    (df_bookings["booking_date"] >= df_bookings["effective_date"]) &
    ((df_bookings["end_date"].isna()) | (df_bookings["booking_date"] <= df_bookings["end_date"]))
].copy()

# Aggregate bookings per flight
booking_agg = df_bookings.groupby("flight_id").agg(
    total_ticket_revenue   = ("ticket_price", "sum"),
    total_passenger_count  = ("passenger_count", "sum"),
    num_bookings           = ("booking_id", "count")
).reset_index()

# 2. Enrich: Aggregate Sensor Data (average per sensor_type per flight)
sensor_agg = df_sensors.groupby(["flight_id", "sensor_type"])["sensor_value"].mean().unstack()
sensor_agg.reset_index(inplace=True)

# 3. Enrich: Process Weather Events (only those between departure & arrival)
df_weather_events["event_time"] = pd.to_datetime(df_weather_events["event_time"], errors='coerce')
df_flights_times = df_flights[["flight_id", "departure_time", "arrival_time"]]
df_weather_merged = pd.merge(df_flights_times, df_weather_events, on="flight_id", how="inner")
df_weather_merged = df_weather_merged[
    (df_weather_merged["event_time"] >= df_weather_merged["departure_time"]) &
    (df_weather_merged["event_time"] <= df_weather_merged["arrival_time"])
].copy()

weather_agg = df_weather_merged.groupby("flight_id").agg(
    avg_temperature_flight     = ("temperature", "mean"),
    max_temperature_flight     = ("temperature", "max"),
    avg_wind_speed_flight      = ("wind_speed", "mean"),
    max_wind_speed_flight      = ("wind_speed", "max"),
    predominant_wind_direction = ("wind_direction", lambda x: x.mode().iloc[0] if not x.mode().empty else None),
    pct_rain_events            = ("weather_condition", lambda x: np.mean(x == "Rain")),
    num_weather_points         = ("weather_condition", "count")
).reset_index()

# 4. Combine all Silver-level enrichments
df_silver = df_flights.copy()
df_silver = pd.merge(df_silver, booking_agg, on="flight_id", how="left")
df_silver = pd.merge(df_silver, sensor_agg, on="flight_id", how="left")
df_silver = pd.merge(df_silver, weather_agg, on="flight_id", how="left")

# Fill NaN for aggregated fields with 0
for col in ["total_ticket_revenue", "total_passenger_count", "num_bookings"]:
    if col in df_silver:
        df_silver[col] = df_silver[col].fillna(0)

# Create delay_flag (>30 minutes) and on_time_flag (<=15 minutes)
df_silver["delay_flag"] = (df_silver["delay_minutes"] > 30).astype(int)
df_silver["on_time_flag"] = (df_silver["delay_minutes"] <= 15).astype(int)

# Save the Silver layer dataset
df_silver.to_csv("silver/silver_flights.csv", index=False)
print("silver/silver_flights.csv generated.")

#########################################
# GOLD LAYER - AGGREGATED BUSINESS METRICS & ML FEATURE TABLE
#########################################
# In the Gold layer, we compute business metrics and prepare enriched CSVs.

# 1. Gold KPI: Enriched metrics per Airport/Gate
gold_kpis = df_silver.groupby(["airport", "gate"]).agg(
    total_flights            = ("flight_id", "count"),
    avg_delay                = ("delay_minutes", "mean"),
    total_passenger_count    = ("total_passenger_count", "sum"),
    total_ticket_revenue     = ("total_ticket_revenue", "sum"),
    pct_flights_delayed_over_30 = ("delay_flag", "mean"),
    pct_on_time_performance     = ("on_time_flag", "mean"),
    avg_flight_duration      = ("flight_duration", "mean")
).reset_index()

# Derive additional columns
gold_kpis["avg_passengers_per_flight"] = (
    gold_kpis["total_passenger_count"] / gold_kpis["total_flights"]
).round(2)
gold_kpis["avg_revenue_per_flight"] = (
    gold_kpis["total_ticket_revenue"] / gold_kpis["total_flights"]
).round(2)

# Reorder columns for readability
gold_kpis = gold_kpis[[
    "airport", "gate",
    "total_flights", "avg_delay",
    "total_passenger_count", "avg_passengers_per_flight",
    "total_ticket_revenue", "avg_revenue_per_flight",
    "pct_flights_delayed_over_30", "pct_on_time_performance",
    "avg_flight_duration"
]]

gold_kpis.to_csv("gold/gold_kpis.csv", index=False)
print("gold/gold_kpis.csv generated.")

# 2. Gold Time-Series KPI: Daily aggregates
df_daily = df_silver.copy()
df_daily["flight_day"] = df_daily["departure_time"].dt.date
daily_agg = df_daily.groupby("flight_day").agg(
    total_flights        = ("flight_id", "count"),
    avg_delay            = ("delay_minutes", "mean"),
    total_ticket_revenue = ("total_ticket_revenue", "sum"),
    total_passengers     = ("total_passenger_count", "sum"),
    pct_on_time          = ("on_time_flag", "mean")
).reset_index()

daily_agg.to_csv("gold/gold_kpis_daily.csv", index=False)
print("gold/gold_kpis_daily.csv generated.")

# 3. Gold ML Feature Table:
df_silver["hour_of_day"] = df_silver["departure_time"].dt.hour
df_silver["day_of_week"] = df_silver["departure_time"].dt.dayofweek

ml_features = df_silver[[
    "flight_id", "airport", "gate",
    "delay_minutes", "delay_flag", "on_time_flag",
    "hour_of_day", "day_of_week",
    "Engine Temperature", "Fuel Level", "Vibration", "Air Pressure", "Airspeed",
    "avg_temperature_flight", "max_temperature_flight",
    "avg_wind_speed_flight", "max_wind_speed_flight",
    "predominant_wind_direction", "pct_rain_events", "num_weather_points",
    "total_ticket_revenue", "total_passenger_count", "num_bookings",
    "flight_duration"
]]

ml_features.to_csv("gold/gold_ml_features.csv", index=False)
print("gold/gold_ml_features.csv generated.")

#########################################
# OPTIONAL: Create Dashboard Mockup as PNG
#########################################
def create_dashboard(output_file="gold/dashboard_mockup.png"):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create a 3x2 layout for enriched KPIs and BI insights
    fig = make_subplots(
        rows=3, cols=2,
        column_widths=[0.6, 0.4],
        row_heights=[0.3, 0.3, 0.4],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "domain"}],
            [{"type": "xy"}, {"type": "xy"}]
        ],
        subplot_titles=(
            "Avg Delay per Airport/Gate",
            "Avg Passengers per Flight (Sample Gates)",
            "Daily Avg Delay Trend",
            "KPI Indicators",
            "Delay vs. Flight Duration",
            "Alerts & Recommendations"
        )
    )

    # 1. Bar Chart: Avg Delay per Airport/Gate (take top 5 by total_flights)
    sample_ag = gold_kpis.sort_values("total_flights", ascending=False).head(5)
    x_ag = [f"{row['airport']}/{row['gate']}" for _, row in sample_ag.iterrows()]
    y_ag = sample_ag["avg_delay"]
    fig.add_trace(
        go.Bar(x=x_ag, y=y_ag, name="Avg Delay (min)"),
        row=1, col=1
    )

    # 2. Bar Chart: Avg Passengers per Flight (same sample gates)
    y_ppf = sample_ag["avg_passengers_per_flight"]
    fig.add_trace(
        go.Bar(x=x_ag, y=y_ppf, marker_color="indianred", name="Avg Passengers"),
        row=1, col=2
    )

    # 3. Line Chart: Daily Avg Delay Trend (last 10 days)
    daily_sample = daily_agg.sort_values("flight_day").tail(10)
    fig.add_trace(
        go.Scatter(
            x=daily_sample["flight_day"],
            y=daily_sample["avg_delay"],
            mode="lines+markers",
            name="Daily Avg Delay"
        ),
        row=2, col=1
    )

    # 4. KPI Indicators: Total Flights and Total Revenue (latest day)
    total_flights = daily_agg["total_flights"].sum()
    total_revenue = daily_agg["total_ticket_revenue"].sum()
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=total_flights,
            title={"text": "Total Flights", "font": {"size": 16}},
            number={"font": {"size": 32}},
            domain={"x": [0, 0.5], "y": [0, 1]}
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=total_revenue,
            title={"text": "Total Revenue", "font": {"size": 16}},
            number={"font": {"size": 32}},
            domain={"x": [0.5, 1], "y": [0, 1]}
        ),
        row=2, col=2
    )

    # 5. Scatter Chart: Delay vs Flight Duration (200-sample points)
    sample_scatter = df_silver.sample(200, random_state=2)
    fig.add_trace(
        go.Scatter(
            x=sample_scatter["flight_duration"],
            y=sample_scatter["delay_minutes"],
            mode="markers",
            marker=dict(size=8, opacity=0.6),
            name="Delay vs Duration"
        ),
        row=3, col=1
    )

    # 6. Text Panel: Alerts & Recommendations
    fig.add_trace(
        go.Scatter(x=[0], y=[0], mode="markers", marker_opacity=0, showlegend=False),
        row=3, col=2
    )
    fig.update_xaxes(visible=False, row=3, col=2)
    fig.update_yaxes(visible=False, row=3, col=2)
    alerts_text = (
        "<b>Live Alerts & Recommendations</b><br><br>"
        "• Identify gates where avg_delay > 20 min.<br>"
        "• Focus maintenance on flights with flight_duration > 240 min.<br>"
        "• Increase staffing on gates with low on-time performance."
    )
    fig.add_annotation(
        text=alerts_text,
        xref="x domain", yref="y domain",
        x=0, y=1,
        showarrow=False,
        align="left",
        font=dict(size=12),
        row=3, col=2
    )

    # Global Filters Annotation
    fig.add_annotation(
        text="<b>Global Filters:</b> [Date Range]  [Airport ▼]  [Gate ▼]  [Metric ▼]",
        xref="paper", yref="paper",
        x=0.5, y=1.18,
        showarrow=False,
        font=dict(size=14)
    )

    # Global Title & Layout
    fig.update_layout(
        title={
            "text": "Airline Operations & Enriched KPIs Dashboard",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 22}
        },
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="#F9F9F9",
        margin=dict(l=40, r=40, t=140, b=40)
    )

    # Save the dashboard as a PNG file in the gold folder
    fig.write_image(output_file, scale=2)
    print(f"Dashboard mockup saved as {output_file}")

if __name__ == "__main__":
    create_dashboard()
