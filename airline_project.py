import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

# ---------------------------------------------------
# Create directories if they don't exist:
#   - bronze/      (raw ingested data)
#   - silver/      (cleaned & enriched data)
#   - gold/        (aggregated KPIs & ML features)
# ---------------------------------------------------
for folder in ["bronze", "silver", "gold"]:
    os.makedirs(folder, exist_ok=True)

#########################################
# BRONZE LAYER
#########################################

# 1. SCD Type 2: ticketPrices.csv (fare_classes)
fare_records = []
fare_classes = ["Y", "W", "F"]  # Economy, Premium, First
start_base = datetime(2022, 1, 1)
for fc in fare_classes:
    eff = start_base
    # Generate a few historical versions (each ~180 days), then one current
    for version in range(3):
        end = eff + timedelta(days=180)
        base_price = round(random.uniform(100, 500) * (1 + 0.1 * version), 2)
        cancellation_fee = round(random.uniform(20, 100) * (1 + 0.05 * version), 2)
        baggage_allowance = random.choice([20, 30, 40])  # kg
        fare_records.append({
            "fare_class_id": fc,
            "base_price": base_price,
            "cancellation_fee": cancellation_fee,
            "baggage_allowance": baggage_allowance,
            "effective_date": eff.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "current_flag": "N"
        })
        eff = end + timedelta(days=1)
    # Last version marked current
    base_price = round(random.uniform(100, 500) * 1.5, 2)
    cancellation_fee = round(random.uniform(20, 100) * 1.2, 2)
    baggage_allowance = random.choice([20, 30, 40])
    fare_records.append({
        "fare_class_id": fc,
        "base_price": base_price,
        "cancellation_fee": cancellation_fee,
        "baggage_allowance": baggage_allowance,
        "effective_date": eff.strftime("%Y-%m-%d"),
        "end_date": None,
        "current_flag": "Y"
    })

df_fares = pd.DataFrame(fare_records)
df_fares.to_csv("bronze/ticketPrices.csv", index=False)


# 2. Static Dimension: airport_distances.csv
airports = ["JFK", "LAX", "ORD", "ATL", "DFW"]
dist_records = []
for i in range(len(airports)):
    for j in range(i + 1, len(airports)):
        a1 = airports[i]
        a2 = airports[j]
        dist = random.randint(500, 3000)  # miles
        time = round(dist / random.uniform(400, 550) * 60, 1)  # flight time in minutes
        dist_records.append({
            "airport_origin": a1,
            "airport_destination": a2,
            "time_of_flight": time,
            "distance": dist
        })
        # Add reverse direction
        dist_records.append({
            "airport_origin": a2,
            "airport_destination": a1,
            "time_of_flight": time,
            "distance": dist
        })

df_dist = pd.DataFrame(dist_records)
df_dist.to_csv("bronze/airport_distances.csv", index=False)


# 3. Streaming: newflights.csv (new flight entries)
num_flights = 500
flight_ids = [f"F{str(i).zfill(4)}" for i in range(1, num_flights + 1)]
flights = []
start_date = datetime(2023, 1, 1)

for fid in flight_ids:
    dep = start_date + timedelta(
        days=random.randint(0, 364),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )
    origin = random.choice(airports)
    dest = random.choice([a for a in airports if a != origin])
    row = df_dist[
        (df_dist["airport_origin"] == origin) &
        (df_dist["airport_destination"] == dest)
    ].iloc[0]
    scheduled_minutes = row["time_of_flight"]
    delay = random.randint(0, 120)
    arr = dep + timedelta(minutes=scheduled_minutes + delay)
    latitude = round(random.uniform(25.0, 49.0), 4)
    longitude = round(random.uniform(-125.0, -70.0), 4)

    flights.append({
        "flight_id": fid,
        "departure_time": dep.strftime("%Y-%m-%d %H:%M:%S"),
        "arrival_time": arr.strftime("%Y-%m-%d %H:%M:%S"),
        "airport_origin": origin,
        "airport_destination": dest,
        "latitude": latitude,
        "longitude": longitude,
        "flight_date": dep.date().strftime("%Y-%m-%d"),
        "scheduled_minutes": scheduled_minutes,
        "delay_minutes": delay
    })

df_newflights = pd.DataFrame(flights)
df_newflights.to_csv("bronze/newflights.csv", index=False)


# 4. Streaming: bookings.csv (tickets bought for a flight)
num_bookings = 1500
booking_records = []
booking_channels = ["Online", "TravelAgency", "MobileApp"]

for i in range(1, num_bookings + 1):
    booking_id = f"B{str(i).zfill(6)}"
    ticket_id = random.choice(df_fares["fare_class_id"])
    flight_id = random.choice(flight_ids)

    dep_str = df_newflights.loc[
        df_newflights["flight_id"] == flight_id, "departure_time"
    ].values[0]
    dep_dt = datetime.strptime(dep_str, "%Y-%m-%d %H:%M:%S")
    bdate = dep_dt - timedelta(days=random.randint(1, 60))

    passenger_count = random.randint(1, 3)
    channel = random.choice(booking_channels)

    booking_records.append({
        "booking_id": booking_id,
        "ticket_id": ticket_id,
        "flight_id": flight_id,
        "booking_date": bdate.strftime("%Y-%m-%d"),
        "passenger_count": passenger_count,
        "booking_channel": channel
    })

df_bookings = pd.DataFrame(booking_records)
df_bookings.to_csv("bronze/bookings.csv", index=False)


# 5. Additional Source: weather_api.csv (weather snapshots per airport & date)
weather_records = []
conditions = ["Clear", "Sunny", "Cloudy", "Rain", "Storm"]

for date in pd.date_range(start="2023-01-01", end="2023-12-31"):
    for ap in airports:
        weather_records.append({
            "date": date.strftime("%Y-%m-%d"),
            "airport": ap,
            "temperature": round(random.uniform(-10, 35), 1),
            "weather_condition": random.choice(conditions)
        })

df_weather_api = pd.DataFrame(weather_records)
df_weather_api.to_csv("bronze/weather_api.csv", index=False)


#########################################
# SILVER LAYER
#########################################

# 1. Enrich bookings: join with ticketPrices (SCD2)
df_book = pd.read_csv("bronze/bookings.csv")
df_fare = pd.read_csv(
    "bronze/ticketPrices.csv",
    parse_dates=["effective_date", "end_date"]
)
df_book["booking_date"] = pd.to_datetime(df_book["booking_date"])

# Merge fare-classes onto bookings, keeping only the row where booking_date
# ∈ [effective_date, end_date] (or end_date is null)
merged = pd.merge(
    df_book,
    df_fare,
    left_on="ticket_id",
    right_on="fare_class_id",
    how="left"
)
merged = merged[
    (merged["booking_date"] >= merged["effective_date"]) &
    ((merged["end_date"].isna()) | (merged["booking_date"] <= merged["end_date"]))
]
merged.to_csv("silver/bookings_enriched.csv", index=False)


# 2. Flight Delay: compute 'late' flag if delay_minutes > 0
df_flights = pd.read_csv(
    "bronze/newflights.csv",
    parse_dates=["departure_time", "arrival_time"]
)
df_flights["late"] = (df_flights["delay_minutes"] > 0).astype(int)
df_flights.to_csv("silver/flight_delay.csv", index=False)


# 3. Late arrival per airport (daily): total flights, total delay, count late, pct late
grouped = df_flights.groupby(
    ["airport_origin", df_flights["departure_time"].dt.date]
).agg(
    total_flights=("flight_id", "count"),
    total_delay=("delay_minutes", "sum"),
    late_flights=("late", "sum")
).reset_index().rename(columns={"departure_time": "date"})
grouped["pct_late"] = grouped["late_flights"] / grouped["total_flights"]
grouped.to_csv("silver/late_arrival_per_airport.csv", index=False)


#########################################
# GOLD LAYER
#########################################

# 1. KPI: average delay per airport_origin
gold_kpis = df_flights.groupby("airport_origin").agg(
    avg_delay=("delay_minutes", "mean"),
    total_flights=("flight_id", "count"),
    pct_late=("late", "mean")
).reset_index()
gold_kpis.to_csv("gold/gold_kpis.csv", index=False)


# 2. ML Features: combine flights, bookings_enriched, and weather
#   - booking aggregates per flight
bk_agg = merged.groupby("flight_id").agg(
    total_ticket_revenue=("passenger_count", "sum"),   # assume revenue ~ passenger_count * 50
    total_passenger_count=("passenger_count", "sum"),
    num_bookings=("booking_id", "count")
).reset_index()

#   - weather join on flight_date & airport_origin
df_weather = pd.read_csv("bronze/weather_api.csv")
df_flights["flight_date"] = df_flights["departure_time"].dt.date.astype(str)

ml = pd.merge(df_flights, bk_agg, on="flight_id", how="left")
ml = pd.merge(
    ml,
    df_weather,
    left_on=["flight_date", "airport_origin"],
    right_on=["date", "airport"],
    how="left"
)

#   - select relevant features
ml_features = ml[[
    "flight_id",
    "airport_origin",
    "airport_destination",
    "departure_time",
    "arrival_time",
    "delay_minutes",
    "late",
    "temperature",
    "weather_condition",
    "total_ticket_revenue",
    "total_passenger_count",
    "num_bookings"
]]
ml_features.to_csv("gold/gold_ml_features.csv", index=False)

print("Data generation complete.")
