import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# -------------------------------------------------------------------
# Generate Branch Data (Static Dimension)
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Generate Flight Data (Batch)
# -------------------------------------------------------------------

num_flights = 10000  # Change this number for more flights
airports = ["JFK", "LAX", "ORD", "ATL", "DFW"]
branch_list = [b["branch"] for b in branches]

# Generate flight IDs (e.g., F0001, F0002, ..., F1000)
flight_ids = [f"F{str(i).zfill(4)}" for i in range(1, num_flights + 1)]

# Define date range for flight departures
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
total_seconds = int((end_date - start_date).total_seconds())

# Generate random departure times within the year
departure_times = [
    start_date + timedelta(seconds=random.randint(0, total_seconds))
    for _ in range(num_flights)
]

# Generate random flight durations (between 1 to 5 hours)
durations = [timedelta(hours=random.randint(1, 5)) for _ in range(num_flights)]
arrival_times = [dt + d for dt, d in zip(departure_times, durations)]

# Generate random delay minutes (0 to 60 minutes)
delay_minutes = [random.randint(0, 60) for _ in range(num_flights)]

# Randomly assign airports and branches to each flight
airports_assigned = [random.choice(airports) for _ in range(num_flights)]
branches_assigned = [random.choice(branch_list) for _ in range(num_flights)]
flight_dates = [dt.date() for dt in departure_times]

# Create DataFrame for flights
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
# Each sensor reading includes:
#   - sensor_id: a unique identifier for the sensor reading.
#   - flight_id: the flight to which this sensor reading belongs.
#   - sensor_type: the type of sensor (e.g., Engine Temperature, Fuel Level, Vibration, Air Pressure, Airspeed).
#   - sensor_value: the measured value based on realistic ranges.
#   - event_time: the timestamp when the reading was recorded (randomly between departure and arrival).

# Define the sensor types and realistic ranges for each sensor's value.
sensor_types = [
    {"type": "Engine Temperature", "min": 500, "max": 700},  # Temperature in °C
    {"type": "Fuel Level", "min": 20, "max": 100},             # Fuel level in percentage (%)
    {"type": "Vibration", "min": 0, "max": 5},                 # Vibration amplitude (arbitrary units)
    {"type": "Air Pressure", "min": 90, "max": 110},           # Air pressure in kPa
    {"type": "Airspeed", "min": 300, "max": 600}               # Airspeed in knots
]

sensor_records = []
for fid, dep, arr in zip(df_flights["flight_id"], df_flights["departure_time"], df_flights["arrival_time"]):
    dep_dt = datetime.strptime(dep, "%Y-%m-%d %H:%M:%S")
    arr_dt = datetime.strptime(arr, "%Y-%m-%d %H:%M:%S")
    # For each flight, generate between 1 and 3 readings for each sensor type.
    for sensor in sensor_types:
        num_records = random.randint(1, 3)
        for _ in range(num_records):
            # Create a unique sensor ID by combining sensor type initials and a random number.
            sensor_id = f"{sensor['type'][:2].upper()}{random.randint(1000, 9999)}"
            # Generate a random event time between departure and arrival
            event_time = dep_dt + timedelta(seconds=random.randint(0, int((arr_dt - dep_dt).total_seconds())))
            # Generate a sensor value within the defined range
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

# -------------------------------------------------------------------
# Generate Weather Data (Batch with Late Arrivals)
# -------------------------------------------------------------------
# Generate daily weather data for each airport from 2023-01-01 to 2023-12-31.
date_range = pd.date_range(start="2023-01-01", end="2023-12-31")
weather_conditions = ["Clear", "Sunny", "Cloudy", "Rain", "Storm"]
weather_records = []
for date in date_range:
    for airport in airports:
        temperature = round(random.uniform(5, 30), 1)  # Temperature between 5 and 30 degrees
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
