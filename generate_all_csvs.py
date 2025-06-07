#!/usr/bin/env python3
"""
This script generates synthetic CSV files in the following structure:
  BRONZE/
    flight_routs.csv
    flights.csv
    weather.csv
    tickets_prices.csv
    booked_ticket.csv
    flights_reports.csv

  SILVER/
    silver_flight_routs.csv
    silver_flights.csv
    silver_tickets_prices.csv
    silver_weather.csv
    silver_tickets.csv

  GOLD/
    gold_dim_profit.csv
    gold_dim_flight_info.csv
    gold_dim_delay.csv
    gold_dim_most_popular.csv
    gold_fact_monthly_performance_report.csv
    gold_dim_ml_weather.csv
    gold_dim_ml_routes.csv
    gold_dim_ml_bording_data.csv
    gold_fact_ml_training.csv
    gold_monthly_performance.csv
"""

import os
import uuid
import random
from datetime import datetime, timedelta
import pandas as pd

# -------------------------------------------------
# 0. Create directories if they don't exist
# -------------------------------------------------
for layer in ("BRONZE", "SILVER", "GOLD"):
    os.makedirs(layer, exist_ok=True)

# -------------------------------------------------
# 1. Configuration: how many records to generate
# -------------------------------------------------
NUM_ROUTES = 10
NUM_FLIGHTS = 50
WEATHER_POINTS_PER_FLIGHT = 5
NUM_PRICE_RULES = 8
NUM_BOOKED_TICKETS = 150
NUM_FLIGHT_REPORTS = NUM_FLIGHTS
NUM_SILVER_TICKETS = 120

YEARS = [2023, 2024]
MONTHS = list(range(1, 13))
NUM_GOLD_FACT_MONTHLY = len(YEARS) * len(MONTHS)
NUM_GOLD_ML = 80

# -------------------------------------------------
# 2. Helper lists & functions
# -------------------------------------------------
AIRPORT_CODES = ["JFK", "LAX", "ORD", "DFW", "ATL", "CDG", "LHR", "HND", "SYD", "DXB"]
ORDER_METHODS = ["Online", "Agent", "Mobile"]
TICKET_CLASSES = ["First", "Business", "Economy"]
LUGGAGE_CLASSES = ["None", "CarryOn", "Checked"]
FLIGHT_STATUS = ["On Time", "Delayed", "Cancelled"]
WEATHER_CONDITIONS = ["Sunny", "Cloudy", "Rain", "Snow", "Clear", "Overcast"]
FIRST_NAMES = [
    "Liam", "Olivia", "Noah", "Emma", "Oliver", "Ava", "Elijah", "Sophia",
    "William", "Isabella", "James", "Mia", "Benjamin", "Charlotte"
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez"
]

def pick_route_pair():
    origin, destination = random.sample(AIRPORT_CODES, 2)
    return origin, destination

def random_lat_lon():
    lat = round(random.uniform(-90.0, 90.0), 6)
    lon = round(random.uniform(-180.0, 180.0), 6)
    return lat, lon

def random_datetime(start: datetime, end: datetime) -> datetime:
    delta = end - start
    total_seconds = int(delta.total_seconds())
    rand_offset = random.randint(0, total_seconds)
    return start + timedelta(seconds=rand_offset)

now = datetime.now()

# -------------------------------------------------
# 3. BRONZE‐layer CSVs
# -------------------------------------------------

# 3.1 flight_routs.csv
routes = []
for _ in range(NUM_ROUTES):
    origin, destination = pick_route_pair()
    lat_o, lon_o = random_lat_lon()
    lat_d, lon_d = random_lat_lon()
    time_of_flight = round(random.uniform(1.0, 12.0), 2)
    distance = round(time_of_flight * random.uniform(400, 550), 2)
    routes.append({
        "airport_origin": origin,
        "airport_destination": destination,
        "latitude_origin": lat_o,
        "longitude_origin": lon_o,
        "latitude_destination": lat_d,
        "longitude_destination": lon_d,
        "time_of_flight": time_of_flight,
        "distance": distance
    })
df_routes = pd.DataFrame(routes)
df_routes.to_csv("BRONZE/flight_routs.csv", index=False)

# 3.2 flights.csv
flights = []
for _ in range(NUM_FLIGHTS):
    flight_id = str(uuid.uuid4())
    origin, destination = pick_route_pair()
    est_dep = random_datetime(now - timedelta(days=90), now)
    duration = timedelta(hours=random.uniform(1.0, 12.0))
    est_arr = est_dep + duration
    actual_dep = est_dep + timedelta(minutes=random.randint(-30, 30))
    actual_arr = actual_dep + duration + timedelta(minutes=random.randint(-20, 40))
    avg_height = round(random.uniform(30000, 40000), 2)
    status = random.choice(FLIGHT_STATUS)
    status_change = random_datetime(est_dep, est_arr)
    actual_flag = random.choice([True, False])
    flights.append({
        "flight_id": flight_id,
        "departure_time": est_dep.isoformat(sep=" "),
        "arrival_time": est_arr.isoformat(sep=" "),
        "airport_origin": origin,
        "airport_destination": destination,
        "average_height": avg_height,
        "status": status,
        "status_change_time": status_change.isoformat(sep=" "),
        "actual": actual_flag
    })
df_flights = pd.DataFrame(flights)
df_flights.to_csv("BRONZE/flights.csv", index=False)

# 3.3 weather.csv
weather_rows = []
for flight in flights:
    fid = flight["flight_id"]
    for point_idx in range(WEATHER_POINTS_PER_FLIGHT):
        lat, lon = random_lat_lon()
        height = round(random.uniform(0, 40000), 2)
        temp = round(random.uniform(-50, 40), 2)
        humidity = round(random.uniform(10, 100), 2)
        wind_speed = round(random.uniform(0, 150), 2)
        wind_dir = round(random.uniform(0, 360), 2)
        condition = random.choice(WEATHER_CONDITIONS)
        sample_date = random_datetime(now - timedelta(days=90), now).date().isoformat()
        time_stamp = random_datetime(now - timedelta(days=90), now).isoformat(sep=" ")
        weather_rows.append({
            "flight_id": fid,
            "point_index": point_idx,
            "latitude": lat,
            "longitude": lon,
            "height": height,
            "temperature": temp,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "wind_direction": wind_dir,
            "condition": condition,
            "sample_for_date": sample_date,
            "time_stamp": time_stamp
        })
df_weather = pd.DataFrame(weather_rows)
df_weather.to_csv("BRONZE/weather.csv", index=False)

# 3.4 tickets_prices.csv
price_rules = []
for _ in range(NUM_PRICE_RULES):
    price_id = str(uuid.uuid4())
    origin, destination = pick_route_pair()
    first_price = round(random.uniform(500, 2000), 2)
    business_price = round(first_price * random.uniform(0.6, 0.9), 2)
    economy_price = round(first_price * random.uniform(0.3, 0.6), 2)
    first_lugg = round(random.uniform(50, 150), 2)
    business_lugg = round(random.uniform(40, 120), 2)
    economy_lugg = round(random.uniform(20, 80), 2)
    start = random_datetime(now - timedelta(days=365), now - timedelta(days=180)).date()
    end = start + timedelta(days=random.randint(30, 180))
    actual_flag = random.choice([True, False])
    price_rules.append({
        "price_id": price_id,
        "airport_origin": origin,
        "airport_destination": destination,
        "first_class_price": first_price,
        "business_class_price": business_price,
        "economy_class_price": economy_price,
        "first_class_luggage": first_lugg,
        "business_class_luggage": business_lugg,
        "economy_class_luggage": economy_lugg,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "actual": actual_flag
    })
df_prices = pd.DataFrame(price_rules)
df_prices.to_csv("BRONZE/tickets_prices.csv", index=False)

# 3.5 booked_ticket.csv
booked = []
for _ in range(NUM_BOOKED_TICKETS):
    ticket_id = str(uuid.uuid4())
    flight_choice = random.choice(flights)["flight_id"]
    tclass = random.choice(TICKET_CLASSES)
    lclass = random.choice(LUGGAGE_CLASSES)
    passenger_id = str(uuid.uuid4())
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    order = random.choice(ORDER_METHODS)
    booking = random_datetime(now - timedelta(days=90), now).date().isoformat()
    booked.append({
        "ticket_id": ticket_id,
        "flight_id": flight_choice,
        "ticket_class": tclass,
        "luggage_class": lclass,
        "passenger_id": passenger_id,
        "passenger_first_name": first,
        "passenger_second_name": last,
        "order_method": order,
        "booking_date": booking
    })
df_booked = pd.DataFrame(booked)
df_booked.to_csv("BRONZE/booked_ticket.csv", index=False)

# 3.6 flights_reports.csv
reports = []
for fl in flights:
    fid = fl["flight_id"]
    actual_dep = datetime.fromisoformat(fl["departure_time"]) + timedelta(
        minutes=random.randint(-20, 40)
    )
    actual_arr = actual_dep + timedelta(hours=random.uniform(1, 12)) + timedelta(
        minutes=random.randint(-30, 60)
    )
    delay_min = int((actual_arr - datetime.fromisoformat(fl["arrival_time"])).total_seconds() / 60)
    report_date = random_datetime(now - timedelta(days=30), now).date().isoformat()
    reports.append({
        "flight_id": fid,
        "actual_departure_time": actual_dep.isoformat(sep=" "),
        "actual_arrival_time": actual_arr.isoformat(sep=" "),
        "delay_in_minutes": max(delay_min, 0),
        "report_date": report_date
    })
df_reports = pd.DataFrame(reports)
df_reports.to_csv("BRONZE/flights_reports.csv", index=False)

# -------------------------------------------------
# 4. SILVER‐layer CSVs
# -------------------------------------------------

# 4.1 silver_flight_routs.csv (same as BRONZE/flight_routs.csv)
df_routes.to_csv("SILVER/silver_flight_routs.csv", index=False)

# 4.2 silver_flights.csv
silver_flights = []
for fl in flights:
    fid = fl["flight_id"]
    origin = fl["airport_origin"]
    destination = fl["airport_destination"]
    est_dep = fl["departure_time"]
    est_arr = fl["arrival_time"]
    actual_dep_silver = datetime.fromisoformat(est_dep) + timedelta(minutes=random.randint(-30, 30))
    actual_arr_silver = datetime.fromisoformat(est_arr) + timedelta(minutes=random.randint(-30, 60))
    delay_min = max(int((actual_arr_silver - datetime.fromisoformat(est_arr)).total_seconds() / 60), 0)
    report_date = random_datetime(now - timedelta(days=60), now).date().isoformat()
    num_pass = random.randint(50, 300)
    num_lugg = random.randint(20, 200)
    profit = round(random.uniform(10000, 100000), 2)
    status = random.choice(FLIGHT_STATUS)
    status_change = random_datetime(datetime.fromisoformat(est_dep), datetime.fromisoformat(est_arr)).isoformat(sep=" ")
    actual_flag = random.choice([True, False])
    silver_flights.append({
        "flight_id": fid,
        "airport_origin": origin,
        "airport_destination": destination,
        "estimated_departure_time": est_dep,
        "estimated_arrival_time": est_arr,
        "actual_departure_time": actual_dep_silver.isoformat(sep=" "),
        "actual_arrival_time": actual_arr_silver.isoformat(sep=" "),
        "delay_in_minutes": delay_min,
        "report_date": report_date,
        "number_of_passengers": num_pass,
        "number_of_luggage": num_lugg,
        "total_flight_profit": profit,
        "status": status,
        "status_change_time": status_change,
        "actual": actual_flag
    })
df_silver_flights = pd.DataFrame(silver_flights)
df_silver_flights.to_csv("SILVER/silver_flights.csv", index=False)

# 4.3 silver_tickets_prices.csv (same as BRONZE/tickets_prices.csv)
df_prices.to_csv("SILVER/silver_tickets_prices.csv", index=False)

# 4.4 silver_weather.csv (one aggregated row per flight)
silver_weather = []
for fl in flights:
    fid = fl["flight_id"]
    heights = [round(random.uniform(30000, 40000), 2) for _ in range(WEATHER_POINTS_PER_FLIGHT)]
    temps = [round(random.uniform(-50, 40), 2) for _ in range(WEATHER_POINTS_PER_FLIGHT)]
    hums = [round(random.uniform(10, 100), 2) for _ in range(WEATHER_POINTS_PER_FLIGHT)]
    speeds = [round(random.uniform(0, 150), 2) for _ in range(WEATHER_POINTS_PER_FLIGHT)]
    dirs_ = [round(random.uniform(0, 360), 2) for _ in range(WEATHER_POINTS_PER_FLIGHT)]
    most_common = random.choice(WEATHER_CONDITIONS)
    count_common = random.randint(1, WEATHER_POINTS_PER_FLIGHT)
    silver_weather.append({
        "flight_id": fid,
        "average_height": round(sum(heights) / len(heights), 2),
        "average_temperature": round(sum(temps) / len(temps), 2),
        "average_humidity": round(sum(hums) / len(hums), 2),
        "average_wind_speed": round(sum(speeds) / len(speeds), 2),
        "average_wind_direction": round(sum(dirs_) / len(dirs_), 2),
        "most_common_condition": most_common,
        "number_of_most_common_condition": count_common
    })
df_silver_weather = pd.DataFrame(silver_weather)
df_silver_weather.to_csv("SILVER/silver_weather.csv", index=False)

# 4.5 silver_tickets.csv (subset of booked tickets)
silver_tickets = []
for i in range(NUM_SILVER_TICKETS):
    rec = random.choice(booked)
    silver_tickets.append({
        "ticket_id": rec["ticket_id"],
        "flight_id": rec["flight_id"],
        "passenger_id": rec["passenger_id"],
        "passenger_first_name": rec["passenger_first_name"],
        "passenger_second_name": rec["passenger_second_name"],
        "order_method": rec["order_method"],
        "ticket_price": round(random.uniform(50, 2000), 2),
        "booking_date": rec["booking_date"]
    })
df_silver_tickets = pd.DataFrame(silver_tickets)
df_silver_tickets.to_csv("SILVER/silver_tickets.csv", index=False)

# -------------------------------------------------
# 5. GOLD‐layer CSVs
# -------------------------------------------------

# 5.1 gold_dim_profit.csv
gold_profit = []
for _ in range(NUM_GOLD_FACT_MONTHLY):
    pid = str(uuid.uuid4())
    total_profit = round(random.uniform(50000, 500000), 2)
    profit_per_pass = round(total_profit / random.randint(1000, 5000), 2)
    avg_ticket_price = round(random.uniform(100, 800), 2)
    gold_profit.append({
        "profit_id": pid,
        "total_profit": total_profit,
        "profit_per_passenger": profit_per_pass,
        "average_ticket_price": avg_ticket_price
    })
df_gold_dim_profit = pd.DataFrame(gold_profit)
df_gold_dim_profit.to_csv("GOLD/gold_dim_profit.csv", index=False)

# 5.2 gold_dim_flight_info.csv
gold_flight_info = []
for _ in range(NUM_GOLD_FACT_MONTHLY):
    fid = str(uuid.uuid4())
    total_flights = random.randint(50, 500)
    total_pass = random.randint(5000, 50000)
    avg_pass_per_flight = round(total_pass / total_flights, 2)
    gold_flight_info.append({
        "flight_info_id": fid,
        "total_flights": total_flights,
        "total_passengers_number": total_pass,
        "average_passengers_number_on_flight": avg_pass_per_flight
    })
df_gold_dim_flight_info = pd.DataFrame(gold_flight_info)
df_gold_dim_flight_info.to_csv("GOLD/gold_dim_flight_info.csv", index=False)

# 5.3 gold_dim_delay.csv
gold_delay = []
for _ in range(NUM_GOLD_FACT_MONTHLY):
    did = str(uuid.uuid4())
    avg_del = round(random.uniform(5, 60), 2)
    min_del = round(random.uniform(0, 5), 2)
    max_del = round(random.uniform(60, 180), 2)
    gold_delay.append({
        "delay_id": did,
        "average_delay": avg_del,
        "minimal_delay": min_del,
        "maximal_delay": max_del
    })
df_gold_dim_delay = pd.DataFrame(gold_delay)
df_gold_dim_delay.to_csv("GOLD/gold_dim_delay.csv", index=False)

# 5.4 gold_dim_most_popular.csv
gold_most_pop = []
for _ in range(NUM_GOLD_FACT_MONTHLY):
    mid = str(uuid.uuid4())
    origin = random.choice(AIRPORT_CODES)
    destination = random.choice([c for c in AIRPORT_CODES if c != origin])
    order = random.choice(ORDER_METHODS)
    gold_most_pop.append({
        "metrics_id": mid,
        "most_popular_origin": origin,
        "most_popular_destination": destination,
        "most_popular_order_method": order
    })
df_gold_dim_most_popular = pd.DataFrame(gold_most_pop)
df_gold_dim_most_popular.to_csv("GOLD/gold_dim_most_popular.csv", index=False)

# 5.5 gold_fact_monthly_performance_report.csv
gold_fact_monthly = []
for idx, year in enumerate(YEARS):
    for m in MONTHS:
        i = idx * len(MONTHS) + (m - 1)
        gold_fact_monthly.append({
            "report_id": str(uuid.uuid4()),
            "year": year,
            "month": m,
            "profit_id": gold_profit[i]["profit_id"],
            "flight_info_id": gold_flight_info[i]["flight_info_id"],
            "delay_id": gold_delay[i]["delay_id"],
            "metrics_id": gold_most_pop[i]["metrics_id"]
        })
df_gold_fact_monthly = pd.DataFrame(gold_fact_monthly)
df_gold_fact_monthly.to_csv("GOLD/gold_fact_monthly_performance_report.csv", index=False)

# 5.6 gold_dim_ml_weather.csv
gold_ml_weather = []
for _ in range(NUM_GOLD_ML):
    wid = str(uuid.uuid4())
    avg_h = round(random.uniform(30000, 40000), 2)
    avg_t = round(random.uniform(-50, 40), 2)
    avg_hum = round(random.uniform(10, 100), 2)
    avg_ws = round(random.uniform(0, 150), 2)
    avg_wd = round(random.uniform(0, 360), 2)
    common_cond = random.choice(WEATHER_CONDITIONS)
    num_common = random.randint(1, 10)
    gold_ml_weather.append({
        "weather_id": wid,
        "average_height": avg_h,
        "average_temperature": avg_t,
        "average_humidity": avg_hum,
        "average_wind_speed": avg_ws,
        "average_wind_direction": avg_wd,
        "most_common_condition": common_cond,
        "number_of_most_common_condition": num_common
    })
df_gold_dim_ml_weather = pd.DataFrame(gold_ml_weather)
df_gold_dim_ml_weather.to_csv("GOLD/gold_dim_ml_weather.csv", index=False)

# 5.7 gold_dim_ml_routes.csv
gold_ml_routes = []
for _ in range(NUM_GOLD_ML):
    rid = str(uuid.uuid4())
    origin, dest = pick_route_pair()
    tof = round(random.uniform(1.0, 12.0), 2)
    dist = round(tof * random.uniform(400, 550), 2)
    gold_ml_routes.append({
        "route_id": rid,
        "airport_origin": origin,
        "airport_destination": dest,
        "time_of_flight": tof,
        "distance": dist
    })
df_gold_dim_ml_routes = pd.DataFrame(gold_ml_routes)
df_gold_dim_ml_routes.to_csv("GOLD/gold_dim_ml_routes.csv", index=False)

# 5.8 gold_dim_ml_bording_data.csv
gold_ml_boarding = []
for _ in range(NUM_GOLD_ML):
    bid = str(uuid.uuid4())
    num_pass = random.randint(50, 300)
    num_lugg = random.randint(20, 200)
    gold_ml_boarding.append({
        "boarding_data_id": bid,
        "number_of_passengers": num_pass,
        "number_of_luggage": num_lugg
    })
df_gold_dim_ml_boarding = pd.DataFrame(gold_ml_boarding)
df_gold_dim_ml_boarding.to_csv("GOLD/gold_dim_ml_bording_data.csv", index=False)

# 5.9 gold_fact_ml_training.csv
gold_fact_ml = []
for i in range(NUM_GOLD_ML):
    gold_fact_ml.append({
        "training_set_id": str(uuid.uuid4()),
        "wether_id": gold_ml_weather[i]["weather_id"],
        "route_id": gold_ml_routes[i]["route_id"],
        "boarding_data_id": gold_ml_boarding[i]["boarding_data_id"],
        "delay_in_minutes": round(random.uniform(0, 180), 2)
    })
df_gold_fact_ml = pd.DataFrame(gold_fact_ml)
df_gold_fact_ml.to_csv("GOLD/gold_fact_ml_training.csv", index=False)

# -------------------------------------------------
# 6. gold_monthly_performance.csv for Streamlit
# -------------------------------------------------
gm = df_gold_fact_monthly.copy()
gm = gm.merge(df_gold_dim_profit, on="profit_id", how="left")
gm = gm.merge(df_gold_dim_flight_info, on="flight_info_id", how="left")
gm = gm.merge(df_gold_dim_delay, on="delay_id", how="left")
gm = gm.merge(df_gold_dim_most_popular, on="metrics_id", how="left")

gold_monthly_perf = gm[[
    "year",
    "month",
    "total_flights",
    "total_profit",
    "average_delay",
    "most_popular_destination",
    "most_popular_order_method"
]]
gold_monthly_perf.to_csv("GOLD/gold_monthly_performance.csv", index=False)

print("All CSV files generated successfully under BRONZE/, SILVER/, and GOLD/ folders.")
