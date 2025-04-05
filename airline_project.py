import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# -------------------------------
# 1. Generate Branch Data (Static Dimension)
# -------------------------------
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

# -------------------------------
# 2. Generate Flight Data (Batch)
# -------------------------------
num_flights = 1000  # Adjust to generate more records if needed
airports = ["JFK", "LAX", "ORD", "ATL", "DFW"]
branch_list = [b["branch"] for b in branches]

# Generate flight IDs (e.g., F0001, F0002, ... F1000)
flight_ids = [f"F{str(i).zfill(4)}" for i in range(1, num_flights + 1)]

# Define date range for flight departures (simulate one year of data)
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
total_seconds = int((end_date - start_date).total_seconds())

# Generate random departure times within the year
departure_times = [
    start_date + timedelta(seconds=random.randint(0, total_seconds))
    for _ in range(num_flights)
]

# Generate random flight durations (between 1 and 5 hours)
durations = [timedelta(hours=random.randint(1, 5)) for _ in range(num_flights)]
arrival_times = [dt + d for dt, d in zip(departure_times, durations)]

# Generate random delay minutes (0 to 60)
delay_minutes = [random.randint(0, 60) for _ in range(num_flights)]

# Randomly assign airports and branches for each flight
airports_assigned = [random.choice(airports) for _ in range(num_flights)]
branches_assigned = [random.choice(branch_list) for _ in range(num_flights)]
flight_dates = [dt.date() for dt in departure_times]

df_flights = pd.DataFrame({
    "flight_id": flight_ids,
    "departure_time": [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in departure_times],
    "arrival_time": [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in arrival_times],
    "delay_minutes": delay_minutes,
    "airport": airports_assigned,
    "branch": branches_assigned,
    "flight_date": [d.strftime("%Y-%m-%d") for d in flight_dates]
})
df_flights.to_csv("flights.csv", index=False)
print("flights.csv generated with", num_flights, "records.")

# -------------------------------
# 3. Generate Sensor Data (Simulated Streaming)
# -------------------------------
# For each flight, generate sensor readings for multiple sensor types.
# Each reading includes sensor_id, flight_id, sensor_type, sensor_value, and event_time.

# Define sensor types and realistic ranges:
sensor_types = [
    {"type": "Engine Temperature", "min": 500, "max": 700},  # °C
    {"type": "Fuel Level", "min": 20, "max": 100},             # Percentage
    {"type": "Vibration", "min": 0, "max": 5},                 # Amplitude in arbitrary units
    {"type": "Air Pressure", "min": 90, "max": 110},           # Pressure in kPa
    {"type": "Airspeed", "min": 300, "max": 600}               # Speed in knots
]

sensor_records = []
for fid, dep, arr in zip(df_flights["flight_id"], df_flights["departure_time"], df_flights["arrival_time"]):
    dep_dt = datetime.strptime(dep, "%Y-%m-%d %H:%M:%S")
    arr_dt = datetime.strptime(arr, "%Y-%m-%d %H:%M:%S")
    # For each sensor type, generate between 1 and 3 readings per flight.
    for sensor in sensor_types:
        num_records = random.randint(1, 3)
        for _ in range(num_records):
            # Create a unique sensor ID (using sensor type initials and a random number)
            sensor_id = f"{sensor['type'][:2].upper()}{random.randint(1000, 9999)}"
            # Random event time between departure and arrival
            event_time = dep_dt + timedelta(seconds=random.randint(0, int((arr_dt - dep_dt).total_seconds())))
            # Generate a sensor value within realistic range
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

# -------------------------------
# 4. Generate Weather Data (Synthetic)
# -------------------------------
# For now, we generate realistic synthetic weather data.
# Later, this section can be updated to fetch data from a weather API.
weather_records = []
# Assume weather data for each airport for each day in 2023
date_range = pd.date_range(start="2023-01-01", end="2023-12-31")
weather_conditions = ["Clear", "Sunny", "Cloudy", "Rain", "Storm"]

for date in date_range:
    for airport in airports:
        temperature = round(random.uniform(5, 30), 1)  # Temperature in °C
        condition = random.choice(weather_conditions)
        weather_records.append({
            "flight_date": date.strftime("%Y-%m-%d"),
            "airport": airport,
            "temperature": temperature,
            "weather_condition": condition
        })

df_weather = pd.DataFrame(weather_records)
df_weather.to_csv("weather.csv", index=False)
print("weather.csv generated with", len(df_weather), "records.")

# -------------------------------
# 5. Generate Customer Data (SCD Type 2)
# -------------------------------
# This dimension simulates customer loyalty changes over time.
num_customers = 500
first_names = ["John", "Jane", "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson"]
loyalty_tiers = ["Silver", "Gold", "Platinum"]

customer_records = []
for i in range(1, num_customers + 1):
    customer_id = f"C{str(i).zfill(4)}"
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    
    # 30% chance the customer has a historical change (simulate SCD Type 2)
    if random.random() < 0.3:
        start_date_c = datetime(2021, 1, 1) + timedelta(days=random.randint(0, 365 * 2))
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
        start_date_c = datetime(2021, 1, 1) + timedelta(days=random.randint(0, 365 * 2))
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


def create_plotly_dashboard(output_file="plotly_dashboard_mock.png"):
    """
    Creates a mock airline operations dashboard with placeholder data:
    - Subplot 1 (top-left): Bar chart for "Average Delay per Airport/Branch"
    - Subplot 2 (top-right): KPI-like card for "Total Flights"
    - Subplot 3 (bottom-left): Line chart for "Delay Trend Over Time"
    - Subplot 4 (bottom-right): Alerts/Recommendations text box
    Saves the final figure as a PNG file.
    """

    # Create a 2x2 subplot layout
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
            "Total Flights (KPI Card)",
            "Time-Series Chart: Delay Trend Over Time",
            "Alerts & Recommendations"
        )
    )

    # 1. Avg Delay per Airport/Branch (Bar Chart) - top-left
    airports = ["JFK", "LAX", "ORD", "ATL", "DFW"]
    avg_delay = [12, 20, 15, 25, 10]  # Placeholder data
    fig.add_trace(
        go.Bar(x=airports, y=avg_delay, marker_color="cornflowerblue", name="Avg Delay (min)"),
        row=1, col=1
    )

    # 2. Total Flights (KPI Card) - top-right (use a Pie or Indicator as a placeholder KPI)
    # Here we'll use an Indicator chart to show a single numeric value, e.g., 1200 flights
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=1200,
            title={"text": "Total Flights", "font": {"size": 16}},
            number={"font": {"size": 40}},
            domain={"x": [0, 1], "y": [0, 1]}
        ),
        row=1, col=2
    )

    # 3. Delay Trend Over Time (Line Chart) - bottom-left
    # Placeholder time series data
    days = list(range(1, 11))  # e.g., 10 days
    delay_trend = [15, 18, 13, 20, 22, 19, 14, 17, 21, 16]
    fig.add_trace(
        go.Scatter(x=days, y=delay_trend, mode="lines+markers", line_color="tomato", name="Delay Trend"),
        row=2, col=1
    )

    # 4. Alerts & Recommendations - bottom-right
    # We'll simulate a text-based approach by using an annotation or shape
    # For demonstration, we'll just show some text with bullet points
    alerts_text = (
        "<b>Live Feed of Real-time Sensor Anomalies</b><br>"
        "• Engine Temp Spike on Flight F002 (685°C)<br>"
        "• Fuel Level Low on Flight F015 (22%)<br>"
        "<br>"
        "<b>Actionable Alerts:</b><br>"
        "• Schedule maintenance for Flight F002<br>"
        "• Investigate abnormal vibration on Flight F023<br>"
    )

    # We'll place an invisible scatter and use annotations for text
    fig.add_trace(
        go.Scatter(x=[0], y=[0], mode="markers", marker_opacity=0),
        row=2, col=2
    )
    fig.update_xaxes(visible=False, row=2, col=2)
    fig.update_yaxes(visible=False, row=2, col=2)

    # Add an annotation for the alerts text
    fig.add_annotation(
        text=alerts_text,
        xref="x domain", yref="y domain",
        x=0, y=1,
        showarrow=False,
        align="left",
        font=dict(size=12),
        row=2, col=2
    )

    # Update layout for the entire figure
    fig.update_layout(
        height=800,
        width=1000,
        title_text="Airline Operations Dashboard Mockup",
        title_font_size=20,
        title_x=0.5,
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="#F9F9F9",
        margin=dict(l=50, r=50, t=100, b=50)
    )

    # Save the figure as a PNG file
    fig.write_image(output_file)
    print(f"Dashboard mockup saved as {output_file}")


if __name__ == "__main__":
    create_plotly_dashboard()
