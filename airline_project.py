import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

#########################################
# BRONZE LAYER - RAW DATA INGESTION (Dirty Data)
#########################################
# In the Bronze layer, raw data is ingested exactly as received.
# This includes missing or invalid values (e.g., missing departure times, non-numeric delays).

# 1. Branch Data (assumed relatively clean)
branches = [
    {"branch": "Branch_A", "city": "New York", "state": "NY"},
    {"branch": "Branch_B", "city": "Los Angeles", "state": "CA"},
    {"branch": "Branch_C", "city": "Chicago", "state": "IL"},
    {"branch": "Branch_D", "city": "Atlanta", "state": "GA"},
    {"branch": "Branch_E", "city": "Dallas", "state": "TX"}
]
df_branches = pd.DataFrame(branches)
df_branches.to_csv("branches.csv", index=False)
print("branches.csv generated.")

# 2. Flight Data (with Dirty Data)
num_flights = 1000  # Adjust as needed
airports = ["JFK", "LAX", "ORD", "ATL", "DFW"]
branch_list = [b["branch"] for b in branches]
flight_ids = [f"F{str(i).zfill(4)}" for i in range(1, num_flights + 1)]
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
total_seconds = int((end_date - start_date).total_seconds())

departure_times = []
arrival_times = []
delay_minutes = []
flight_dates = []
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
    
    departure_times.append(dep_str)
    arrival_times.append(arr.strftime("%Y-%m-%d %H:%M:%S"))
    delay_minutes.append(delay)
    flight_dates.append(dep.date().strftime("%Y-%m-%d"))

df_flights = pd.DataFrame({
    "flight_id": flight_ids,
    "departure_time": departure_times,
    "arrival_time": arrival_times,
    "delay_minutes": delay_minutes,
    "airport": [random.choice(airports) for _ in range(num_flights)],
    "branch": [random.choice(branch_list) for _ in range(num_flights)],
    "flight_date": flight_dates
})
df_flights.to_csv("flights.csv", index=False)
print("flights.csv generated with", num_flights, "records.")

# 3. Sensor Data (Simulated Streaming with Dirty Data)
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
df_sensors.to_csv("sensors.csv", index=False)
print("sensors.csv generated with", len(df_sensors), "records.")

# 4. Weather Data (Synthetic with Dirty Data)
weather_records = []
date_range = pd.date_range(start="2023-01-01", end="2023-12-31")
weather_conditions = ["Clear", "Sunny", "Cloudy", "Rain", "Storm"]
for date in date_range:
    for airport in airports:
        temperature = round(random.uniform(5, 30), 1)
        condition = random.choice(weather_conditions)
        # 3% chance: missing temperature (None)
        if random.random() < 0.03:
            temperature = None
        weather_records.append({
            "flight_date": date.strftime("%Y-%m-%d"),
            "airport": airport,
            "temperature": temperature,
            "weather_condition": condition
        })
df_weather = pd.DataFrame(weather_records)
df_weather.to_csv("weather.csv", index=False)
print("weather.csv generated with", len(df_weather), "records.")

# 5. Customer Data (SCD Type 2) - Assumed Clean
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
df_customers.to_csv("customers.csv", index=False)
print("customers.csv generated with", len(df_customers), "records.")

#########################################
# SILVER LAYER - CLEANED & ENRICHED DATA
#########################################
# In the Silver layer, the raw Bronze data is cleaned, standardized, and enriched.

# Load Bronze CSV files
df_flights = pd.read_csv("flights.csv")
df_branches = pd.read_csv("branches.csv")
df_weather = pd.read_csv("weather.csv")
df_sensors = pd.read_csv("sensors.csv")

# Standardize and Convert Data Types in Flights Data
df_flights["departure_time"] = pd.to_datetime(df_flights["departure_time"], errors='coerce')
df_flights["arrival_time"] = pd.to_datetime(df_flights["arrival_time"], errors='coerce')
df_flights["flight_date"] = pd.to_datetime(df_flights["flight_date"], errors='coerce').dt.date
# Drop rows with invalid/missing key date/time values
df_flights = df_flights.dropna(subset=["departure_time", "arrival_time", "flight_date"])
# Convert delay_minutes to numeric and drop invalid rows
df_flights["delay_minutes"] = pd.to_numeric(df_flights["delay_minutes"], errors='coerce')
df_flights = df_flights.dropna(subset=["delay_minutes"])

# Enrich: Merge flights with branch details (join on "branch")
df_silver = pd.merge(df_flights, df_branches, on="branch", how="left")

# Enrich: Merge flights with weather data (join on flight_date and airport)
df_weather["flight_date"] = pd.to_datetime(df_weather["flight_date"], errors='coerce').dt.date
df_silver = pd.merge(df_silver, df_weather, on=["flight_date", "airport"], how="left")

# Enrich: Aggregate Sensor Data - Compute average sensor_value for each sensor type per flight
sensor_agg = df_sensors.groupby(["flight_id", "sensor_type"])["sensor_value"].mean().unstack()
sensor_agg.reset_index(inplace=True)
# Merge sensor aggregates into the Silver flight data
df_silver = pd.merge(df_silver, sensor_agg, on="flight_id", how="left")

# Save the Silver layer dataset
df_silver.to_csv("silver_flights.csv", index=False)
print("silver_flights.csv generated.")

#########################################
# GOLD LAYER - AGGREGATED BUSINESS METRICS & ML FEATURE TABLE
#########################################
# In the Gold layer, we compute business metrics and prepare an ML feature table.

# Gold KPI: Compute Average Delay per Airport/Branch and Total Flights
gold_kpis = df_silver.groupby(["airport", "branch"]).agg(
    avg_delay=("delay_minutes", "mean"),
    total_flights=("flight_id", "count")
).reset_index()
gold_kpis.to_csv("gold_kpis.csv", index=False)
print("gold_kpis.csv generated.")

# Gold ML Feature Table:
# Derive additional time-based features from departure_time.
df_silver["hour_of_day"] = df_silver["departure_time"].dt.hour
df_silver["day_of_week"] = df_silver["departure_time"].dt.dayofweek

# Select key columns for the ML feature table.
# Ensure the following columns appear exactly as required:
# flight_id, airport, branch, delay_minutes, hour_of_day, day_of_week,
# Engine Temperature, Fuel Level, Vibration, Air Pressure, Airspeed, temperature, weather_condition
ml_features = df_silver[[
    "flight_id", "airport", "branch", "delay_minutes", "hour_of_day", "day_of_week",
    "Engine Temperature", "Fuel Level", "Vibration", "Air Pressure", "Airspeed",
    "temperature", "weather_condition"
]]
ml_features.to_csv("gold_ml_features.csv", index=False)
print("gold_ml_features.csv generated.")


import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_dashboard(output_file="dashboard_mockup.png"):
    # Create a 2x2 subplot layout.
    # - Row 1, Col 1: XY type for a bar chart (Avg Delay per Airport/Branch)
    # - Row 1, Col 2: Domain type for KPI indicators (split side-by-side)
    # - Row 2, Col 1: XY type for a line chart (Delay Trend Over Time)
    # - Row 2, Col 2: XY type for a text-based Alerts & Recommendations panel
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.6, 0.4],
        row_heights=[0.55, 0.45],
        specs=[
            [{"type": "xy"}, {"type": "domain"}],
            [{"type": "xy"}, {"type": "xy"}]
        ],
        subplot_titles=(
            "Avg Delay per Airport/Branch",
            "KPI Indicators",
            "Delay Trend Over Time",
            "Alerts & Recommendations"
        )
    )

    # -----------------------------
    # 1. Bar Chart - Avg Delay per Airport/Branch
    # -----------------------------
    airports = ["JFK", "LAX", "ORD", "ATL", "DFW"]
    avg_delay = [12, 18, 15, 22, 10]  # sample average delay in minutes
    fig.add_trace(
        go.Bar(
            x=airports,
            y=avg_delay,
            marker_color="royalblue",
            name="Avg Delay (min)"
        ),
        row=1, col=1
    )

    # -----------------------------
    # 2. KPI Indicators - Total Flights and Sensor Anomalies
    # -----------------------------
    # Using a "domain" subplot split horizontally into two sections.
    # Indicator for Total Flights (left half)
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=1200,  # sample total flights
            title={"text": "Total Flights", "font": {"size": 14}},
            number={"font": {"size": 32}},
            domain={"x": [0, 0.5], "y": [0, 1]}
        ),
        row=1, col=2
    )
    # Indicator for Sensor Anomalies (right half)
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
    # -----------------------------
    days = list(range(1, 11))  # e.g., 10 days placeholder
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
    # -----------------------------
    # Use an invisible scatter trace to create a placeholder for text.
    fig.add_trace(
        go.Scatter(x=[0], y=[0], mode="markers", marker_opacity=0, showlegend=False),
        row=2, col=2
    )
    # Hide axes in this subplot
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
    # Global Filters Annotation (Displayed Above the Dashboard)
    # -----------------------------
    fig.add_annotation(
        text="<b>Global Filters:</b> [Date Range]  [Airport ▼]  [Branch ▼]  [Flight Status ▼]",
        xref="paper", yref="paper",
        x=0.5, y=1.15,
        showarrow=False,
        font=dict(size=14)
    )

    # -----------------------------
    # Global Title & Layout
    # -----------------------------
    fig.update_layout(
        title={"text": "Airline Operations Dashboard", "x": 0.5, "xanchor": "center", "font": {"size": 20}},
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="#F9F9F9",
        margin=dict(l=50, r=50, t=100, b=50)
    )

    # Save the dashboard as a PNG file
    fig.write_image(output_file, scale=2)
    print(f"Dashboard mockup saved as {output_file}")

if __name__ == "__main__":
    create_dashboard()
