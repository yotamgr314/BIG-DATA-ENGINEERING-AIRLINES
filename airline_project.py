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

# Derive flight_date from departure_time (so we can still use it if needed downstream)
df_flights["flight_date"] = df_flights["departure_time"].dt.date

# At this point, df_flights has columns:
# [flight_id, departure_time, arrival_time, delay_minutes, airport, gate, flight_date]

# Enrich: Link flights to bookings (Data Mart)
# Convert booking_date to datetime.date
df_bookings["booking_date"] = pd.to_datetime(df_bookings["booking_date"], errors='coerce').dt.date

# Convert effective_date and end_date in customers to datetime.date
df_customers["effective_date"] = pd.to_datetime(df_customers["effective_date"], errors='coerce').dt.date
df_customers["end_date"]       = pd.to_datetime(df_customers["end_date"], errors='coerce').dt.date

# Merge bookings with customer loyalty_tier based on booking_date between effective_date and end_date
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

# Merge booking aggregates into Silver
df_silver = pd.merge(df_flights, booking_agg, on="flight_id", how="left")

# Enrich: Aggregate Sensor Data - Compute average sensor_value for each sensor type per flight
sensor_agg = df_sensors.groupby(["flight_id", "sensor_type"])["sensor_value"].mean().unstack()
sensor_agg.reset_index(inplace=True)
# Merge sensor aggregates into the Silver flight data
df_silver = pd.merge(df_silver, sensor_agg, on="flight_id", how="left")

# Enrich: Process Weather Events - Convert types
df_weather_events["event_time"] = pd.to_datetime(df_weather_events["event_time"], errors='coerce')
# Join flights with weather events where event_time between departure and arrival
df_flights_times = df_silver[["flight_id", "departure_time", "arrival_time"]]
df_weather_merged = pd.merge(df_flights_times, df_weather_events, on="flight_id", how="inner")
df_weather_merged = df_weather_merged[
    (df_weather_merged["event_time"] >= df_weather_merged["departure_time"]) &
    (df_weather_merged["event_time"] <= df_weather_merged["arrival_time"])
].copy()

# Aggregate weather events per flight
weather_agg = df_weather_merged.groupby("flight_id").agg(
    avg_temperature_flight     = ("temperature", "mean"),
    max_temperature_flight     = ("temperature", "max"),
    avg_wind_speed_flight      = ("wind_speed", "mean"),
    max_wind_speed_flight      = ("wind_speed", "max"),
    predominant_wind_direction = ("wind_direction", lambda x: x.mode().iloc[0] if not x.mode().empty else None),
    pct_rain_events            = ("weather_condition", lambda x: np.mean(x == "Rain")),
    num_weather_points         = ("weather_condition", "count")
).reset_index()

# Merge weather aggregates into Silver
df_silver = pd.merge(df_silver, weather_agg, on="flight_id", how="left")

# Save the Silver layer dataset
df_silver.to_csv("silver/silver_flights.csv", index=False)
print("silver/silver_flights.csv generated.")

#########################################
# GOLD LAYER - AGGREGATED BUSINESS METRICS & ML FEATURE TABLE
#########################################
# In the Gold layer, we compute business metrics and prepare an ML feature table.

# Gold KPI: Compute Average Delay per Airport/Gate and Total Flights
gold_kpis = df_silver.groupby(["airport", "gate"]).agg(
    avg_delay     = ("delay_minutes", "mean"),
    total_flights = ("flight_id", "count")
).reset_index()
gold_kpis.to_csv("gold/gold_kpis.csv", index=False)
print("gold/gold_kpis.csv generated.")

# Gold ML Feature Table:
# Derive additional time-based features from departure_time.
df_silver["hour_of_day"] = df_silver["departure_time"].dt.hour
df_silver["day_of_week"] = df_silver["departure_time"].dt.dayofweek
df_silver["delay_flag"]  = (df_silver["delay_minutes"] > 30).astype(int)

# Select key columns for the ML feature table.
ml_features = df_silver[[
    "flight_id", "airport", "gate", "delay_minutes", "delay_flag",
    "hour_of_day", "day_of_week",
    "Engine Temperature", "Fuel Level", "Vibration", "Air Pressure", "Airspeed",
    "avg_temperature_flight", "max_temperature_flight",
    "avg_wind_speed_flight", "max_wind_speed_flight",
    "predominant_wind_direction", "pct_rain_events", "num_weather_points",
    "total_ticket_revenue", "total_passenger_count", "num_bookings"
]]
ml_features.to_csv("gold/gold_ml_features.csv", index=False)
print("gold/gold_ml_features.csv generated.")

#########################################
# OPTIONAL: Create Dashboard Mockup as PNG
#########################################
def create_dashboard(output_file="gold/dashboard_mockup.png"):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create a 2x2 subplot layout.
    # - Row 1, Col 1: Bar chart (Avg Delay per Airport/Gate)
    # - Row 1, Col 2: KPI indicators (Total Flights, Sensor Anomalies)
    # - Row 2, Col 1: Line chart (Delay Trend Over Time)
    # - Row 2, Col 2: Text-based Alerts & Recommendations panel
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.6, 0.4],
        row_heights=[0.55, 0.45],
        specs=[
            [{"type": "xy"}, {"type": "domain"}],
            [{"type": "xy"}, {"type": "xy"}]
        ],
        subplot_titles=(
            "Avg Delay per Airport/Gate",
            "KPI Indicators",
            "Delay Trend Over Time",
            "Alerts & Recommendations"
        )
    )

    # -----------------------------
    # 1. Bar Chart - Avg Delay per Airport/Gate
    airports_plot = ["JFK", "LAX", "ORD", "ATL", "DFW"]
    gates_plot    = ["JFK_G1", "LAX_G1", "ORD_G1", "ATL_G1", "DFW_G1"]  # sample gates
    avg_delay_plot = [12, 18, 15, 22, 10]  # sample average delay
    fig.add_trace(
        go.Bar(
            x=[f"{a}/{g}" for a, g in zip(airports_plot, gates_plot)],
            y=avg_delay_plot,
            marker_color="royalblue",
            name="Avg Delay (min)"
        ),
        row=1, col=1
    )

    # -----------------------------
    # 2. KPI Indicators - Total Flights and Sensor Anomalies
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=gold_kpis["total_flights"].sum(),
            title={"text": "Total Flights", "font": {"size": 14}},
            number={"font": {"size": 32}},
            domain={"x": [0, 0.5], "y": [0, 1]}
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=7,  # sample sensor anomalies count
            title={"text": "Sensor Anomalies", "font": {"size": 14}},
            number={"font": {"size": 32}},
            domain={"x": [0.5, 1], "y": [0, 1]}
        ),
        row=1, col=2
    )

    # -----------------------------
    # 3. Line Chart - Delay Trend Over Time
    days = list(range(1, 11))
    delay_trend = [15, 18, 13, 20, 22, 19, 14, 17, 21, 16]
    fig.add_trace(
        go.Scatter(
            x=days,
            y=delay_trend,
            mode="lines+markers",
            line=dict(color="firebrick"),
            name="Delay Trend"
        ),
        row=2, col=1
    )

    # -----------------------------
    # 4. Text Panel - Alerts & Recommendations
    fig.add_trace(
        go.Scatter(x=[0], y=[0], mode="markers", marker_opacity=0, showlegend=False),
        row=2, col=2
    )
    fig.update_xaxes(visible=False, row=2, col=2)
    fig.update_yaxes(visible=False, row=2, col=2)
    alerts_text = (
        "<b>Live Alerts & Recommendations</b><br><br>"
        "• Flight F002: Engine Temp Spike (685°C).<br>"
        "&nbsp;&nbsp;Action: Schedule maintenance.<br><br>"
        "• Flight F015: Low Fuel Level (22%).<br>"
        "&nbsp;&nbsp;Action: Verify refueling.<br><br>"
        "• Flight F023: Abnormal Vibration detected.<br>"
        "&nbsp;&nbsp;Action: Inspect engine balance."
    )
    fig.add_annotation(
        text=alerts_text,
        xref="x domain", yref="y domain",
        x=0, y=1,
        showarrow=False,
        align="left",
        font=dict(size=12),
        row=2, col=2
    )

    # -----------------------------
    # Global Filters Annotation
    fig.add_annotation(
        text="<b>Global Filters:</b> [Date Range]  [Airport ▼]  [Gate ▼]  [Flight Status ▼]",
        xref="paper", yref="paper",
        x=0.5, y=1.15,
        showarrow=False,
        font=dict(size=14)
    )

    # -----------------------------
    # Global Title & Layout
    fig.update_layout(
        title={
            "text": "Airline Operations Dashboard",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20}
        },
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="#F9F9F9",
        margin=dict(l=50, r=50, t=120, b=50)
    )

    # Save the dashboard as a PNG file in the gold folder
    fig.write_image(output_file, scale=2)
    print(f"Dashboard mockup saved as {output_file}")

if __name__ == "__main__":
    create_dashboard()
