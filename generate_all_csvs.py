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

# 0. Create directories
for layer in ("BRONZE", "SILVER", "GOLD"):
    os.makedirs(layer, exist_ok=True)

# 1. Configuration
NUM_ROUTES            = 10
NUM_FLIGHTS           = 50
WEATHER_POINTS_PER_FLIGHT = 5
NUM_PRICE_RULES       = 8
NUM_BOOKED_TICKETS    = 150
NUM_SILVER_TICKETS    = 120

YEARS                 = [2023, 2024]
MONTHS                = list(range(1, 13))
NUM_GOLD_FACT_MONTHLY = len(YEARS) * len(MONTHS)
NUM_GOLD_ML           = 80

BOOKING_START = datetime(YEARS[0], 1, 1)
BOOKING_END   = datetime(YEARS[-1], 12, 31)
now           = datetime.now()

# 2. Helpers
AIRPORT_CODES     = ["JFK","LAX","ORD","DFW","ATL","CDG","LHR","HND","SYD","DXB"]
ORDER_METHODS     = ["Online","Agent","Mobile"]
TICKET_CLASSES    = ["Economy","Business","Premium"]
LUGGAGE_CLASSES   = ["None","CarryOn","Checked"]
FLIGHT_STATUS     = ["On Time","Delayed","Cancelled"]
WEATHER_CONDITIONS= ["Sunny","Cloudy","Rain","Snow","Clear","Overcast"]
FIRST_NAMES       = ["Liam","Olivia","Noah","Emma","Oliver","Ava","Elijah","Sophia","William","Isabella","James","Mia","Benjamin","Charlotte"]
LAST_NAMES        = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez","Hernandez","Lopez"]

def pick_route_pair():
    return random.sample(AIRPORT_CODES, 2)

def random_lat_lon():
    return round(random.uniform(-90,90),6), round(random.uniform(-180,180),6)

def random_datetime(start, end):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))

# -------------------------------------------------
# 3. BRONZE‐layer CSVs
# -------------------------------------------------

# 3.1 flight_routs.csv
routes = []
for _ in range(NUM_ROUTES):
    o,d = pick_route_pair()
    lo,la = random_lat_lon()
    do,da = random_lat_lon()
    tf = round(random.uniform(1,12),2)
    dist = round(tf * random.uniform(400,550),2)
    routes.append({
        "airport_origin": o,
        "airport_destination": d,
        "latitude_origin": lo,
        "longitude_origin": la,
        "latitude_destination": do,
        "longitude_destination": da,
        "time_of_flight": tf,
        "distance": dist
    })
df_routes = pd.DataFrame(routes)
df_routes.to_csv("BRONZE/flight_routs.csv", index=False)

# 3.2 flights.csv
flights = []
for _ in range(NUM_FLIGHTS):
    fid = str(uuid.uuid4())
    o,d = pick_route_pair()
    est_dep = random_datetime(now - timedelta(days=90), now)
    dur     = timedelta(hours=random.uniform(1,12))
    est_arr = est_dep + dur
    act_dep = est_dep + timedelta(minutes=random.randint(-30,30))
    act_arr = act_dep + dur + timedelta(minutes=random.randint(-20,40))
    flights.append({
        "flight_id": fid,
        "departure_time": est_dep.isoformat(sep=" "),
        "arrival_time": est_arr.isoformat(sep=" "),
        "airport_origin": o,
        "airport_destination": d,
        "average_height": round(random.uniform(30000,40000),2),
        "status": random.choice(FLIGHT_STATUS),
        "status_change_time": random_datetime(est_dep,est_arr).isoformat(sep=" "),
        "actual": random.choice([True,False])
    })
df_flights = pd.DataFrame(flights)
df_flights.to_csv("BRONZE/flights.csv", index=False)

# 3.3 weather.csv
weather_rows = []
for fl in flights:
    fid = fl["flight_id"]
    for i in range(WEATHER_POINTS_PER_FLIGHT):
        ts = random_datetime(now - timedelta(days=90), now)
        weather_rows.append({
            "flight_id": fid,
            "point_index": i,
            "latitude": random_lat_lon()[0],
            "longitude": random_lat_lon()[1],
            "height": round(random.uniform(0,40000),2),
            "temperature": round(random.uniform(-50,40),2),
            "humidity": round(random.uniform(10,100),2),
            "wind_speed": round(random.uniform(0,150),2),
            "wind_direction": round(random.uniform(0,360),2),
            "condition": random.choice(WEATHER_CONDITIONS),
            "sample_for_date": ts.date().isoformat(),
            "time_stamp": ts.isoformat(sep=" ")
        })
df_weather = pd.DataFrame(weather_rows)
df_weather.to_csv("BRONZE/weather.csv", index=False)

# 3.4 tickets_prices.csv
price_rules = []
for _ in range(NUM_PRICE_RULES):
    pid = str(uuid.uuid4())
    o,d = pick_route_pair()
    fp = round(random.uniform(500,2000),2)
    bp = round(fp*random.uniform(0.6,0.9),2)
    ep = round(fp*random.uniform(0.3,0.6),2)
    start = random_datetime(now - timedelta(days=365), now - timedelta(days=180)).date()
    end   = start + timedelta(days=random.randint(30,180))
    price_rules.append({
        "price_id": pid,
        "airport_origin": o,
        "airport_destination": d,
        "first_class_price": fp,
        "business_class_price": bp,
        "economy_class_price": ep,
        "first_class_luggage": round(random.uniform(50,150),2),
        "business_class_luggage": round(random.uniform(40,120),2),
        "economy_class_luggage": round(random.uniform(20,80),2),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "actual": random.choice([True,False])
    })
df_prices = pd.DataFrame(price_rules)
df_prices.to_csv("BRONZE/tickets_prices.csv", index=False)

# 3.5 booked_ticket.csv
booked = []
for _ in range(NUM_BOOKED_TICKETS):
    tid = str(uuid.uuid4())
    flid= random.choice(flights)["flight_id"]
    booked.append({
        "ticket_id": tid,
        "flight_id": flid,
        "ticket_class": random.choice(TICKET_CLASSES),
        "luggage_class": random.choice(LUGGAGE_CLASSES),
        "passenger_id": str(uuid.uuid4()),
        "passenger_first_name": random.choice(FIRST_NAMES),
        "passenger_second_name": random.choice(LAST_NAMES),
        "order_method": random.choice(ORDER_METHODS),
        "booking_date": random_datetime(BOOKING_START,BOOKING_END).date().isoformat()
    })
df_booked = pd.DataFrame(booked)
df_booked.to_csv("BRONZE/booked_ticket.csv", index=False)

# 3.6 flights_reports.csv
reports = []
for fl in flights:
    fid = fl["flight_id"]
    ad  = random_datetime(now - timedelta(days=30), now)
    aa  = ad + timedelta(hours=random.uniform(1,12)) + timedelta(minutes=random.randint(-30,60))
    reports.append({
        "flight_id": fid,
        "actual_departure_time": ad.isoformat(sep=" "),
        "actual_arrival_time": aa.isoformat(sep=" "),
        "delay_in_minutes": max(int((aa - datetime.fromisoformat(fl["arrival_time"])).total_seconds()/60),0),
        "report_date": random_datetime(now - timedelta(days=30), now).date().isoformat()
    })
df_reports = pd.DataFrame(reports)
df_reports.to_csv("BRONZE/flights_reports.csv", index=False)

# -------------------------------------------------
# 4. SILVER‐layer CSVs
# -------------------------------------------------

# 4.1 flight routes
df_routes.to_csv("SILVER/silver_flight_routs.csv", index=False)

# 4.2 flights
silver_flights = []
for fl in flights:
    est_dep = fl["departure_time"]
    est_arr = fl["arrival_time"]
    ad      = datetime.fromisoformat(est_dep) + timedelta(minutes=random.randint(-30,30))
    aa      = datetime.fromisoformat(est_arr) + timedelta(minutes=random.randint(-30,60))
    silver_flights.append({
        "flight_id": fl["flight_id"],
        "airport_origin": fl["airport_origin"],
        "airport_destination": fl["airport_destination"],
        "estimated_departure_time": est_dep,
        "estimated_arrival_time": est_arr,
        "actual_departure_time": ad.isoformat(sep=" "),
        "actual_arrival_time": aa.isoformat(sep=" "),
        "delay_in_minutes": max(int((aa - datetime.fromisoformat(est_arr)).total_seconds()/60),0),
        "report_date": random_datetime(now - timedelta(days=60), now).date().isoformat(),
        "number_of_passengers": random.randint(50,300),
        "number_of_luggage": random.randint(20,200),
        "total_flight_profit": round(random.uniform(10000,100000),2),
        "status": random.choice(FLIGHT_STATUS),
        "status_change_time": random_datetime(datetime.fromisoformat(est_dep),datetime.fromisoformat(est_arr)).isoformat(sep=" "),
        "actual": random.choice([True,False])
    })
pd.DataFrame(silver_flights).to_csv("SILVER/silver_flights.csv", index=False)

# 4.3 tickets_prices
df_prices.to_csv("SILVER/silver_tickets_prices.csv", index=False)

# 4.4 weather
silver_weather = []
for fl in flights:
    vals = [random.uniform(0,40000) for _ in range(WEATHER_POINTS_PER_FLIGHT)]
    temps= [random.uniform(-50,40) for _ in range(WEATHER_POINTS_PER_FLIGHT)]
    hums = [random.uniform(10,100) for _ in range(WEATHER_POINTS_PER_FLIGHT)]
    spd  = [random.uniform(0,150) for _ in range(WEATHER_POINTS_PER_FLIGHT)]
    wd   = [random.uniform(0,360) for _ in range(WEATHER_POINTS_PER_FLIGHT)]
    silver_weather.append({
        "flight_id": fl["flight_id"],
        "average_height": round(sum(vals)/len(vals),2),
        "average_temperature": round(sum(temps)/len(temps),2),
        "average_humidity": round(sum(hums)/len(hums),2),
        "average_wind_speed": round(sum(spd)/len(spd),2),
        "average_wind_direction": round(sum(wd)/len(wd),2),
        "most_common_condition": random.choice(WEATHER_CONDITIONS),
        "number_of_most_common_condition": random.randint(1,WEATHER_POINTS_PER_FLIGHT)
    })
pd.DataFrame(silver_weather).to_csv("SILVER/silver_weather.csv", index=False)

# 4.5 silver_tickets.csv
# ——————————————
# Now **including** ticket_class (was missing)
silver_tickets = []
for rec in booked:
    silver_tickets.append({
        "ticket_id": rec["ticket_id"],
        "flight_id": rec["flight_id"],
        "ticket_class": rec["ticket_class"],       # <-- added
        "passenger_id": rec["passenger_id"],
        "passenger_first_name": rec["passenger_first_name"],
        "passenger_second_name": rec["passenger_second_name"],
        "order_method": rec["order_method"],
        "ticket_price": round(random.uniform(50,2000),2),
        "booking_date": rec["booking_date"]
    })
# sample a subset
df_silver_tix = pd.DataFrame(silver_tickets).sample(NUM_SILVER_TICKETS)
df_silver_tix.to_csv("SILVER/silver_tickets.csv", index=False)

# -------------------------------------------------
# 5. GOLD‐layer CSVs
# -------------------------------------------------

# 5.1 gold_dim_profit.csv
gold_profit = []
for _ in range(NUM_GOLD_FACT_MONTHLY):
    pid = str(uuid.uuid4())
    tp  = round(random.uniform(50000,500000),2)
    gold_profit.append({
        "profit_id": pid,
        "total_profit": tp,
        "profit_per_passenger": round(tp/random.randint(1000,5000),2),
        "average_ticket_price": round(random.uniform(100,800),2)
    })
df_gp = pd.DataFrame(gold_profit)
df_gp.to_csv("GOLD/gold_dim_profit.csv", index=False)

# 5.2 gold_dim_flight_info.csv
gold_flight_info = []
for _ in range(NUM_GOLD_FACT_MONTHLY):
    fid= str(uuid.uuid4())
    tf = random.randint(50,500)
    tp = random.randint(5000,50000)
    gold_flight_info.append({
        "flight_info_id": fid,
        "total_flights": tf,
        "total_passengers_number": tp,
        "average_passengers_number_on_flight": round(tp/tf,2)
    })
df_gfi = pd.DataFrame(gold_flight_info)
df_gfi.to_csv("GOLD/gold_dim_flight_info.csv", index=False)

# 5.3 gold_dim_delay.csv
gold_delay = []
for _ in range(NUM_GOLD_FACT_MONTHLY):
    did=str(uuid.uuid4())
    gold_delay.append({
        "delay_id": did,
        "average_delay": round(random.uniform(5,60),2),
        "minimal_delay": round(random.uniform(0,5),2),
        "maximal_delay": round(random.uniform(60,180),2)
    })
df_gd = pd.DataFrame(gold_delay)
df_gd.to_csv("GOLD/gold_dim_delay.csv", index=False)

# 5.4 gold_dim_most_popular.csv
gold_most_pop=[]
for _ in range(NUM_GOLD_FACT_MONTHLY):
    mid=str(uuid.uuid4())
    o,d=pick_route_pair()
    gold_most_pop.append({
        "metrics_id": mid,
        "most_popular_origin": o,
        "most_popular_destination": d,
        "most_popular_order_method": random.choice(ORDER_METHODS)
    })
df_gmp=pd.DataFrame(gold_most_pop)
df_gmp.to_csv("GOLD/gold_dim_most_popular.csv", index=False)

# 5.5 gold_fact_monthly_performance_report.csv
gold_fact=[]
for yi,yr in enumerate(YEARS):
    for m in MONTHS:
        idx=yi*len(MONTHS)+(m-1)
        gold_fact.append({
            "report_id": str(uuid.uuid4()),
            "year": yr,
            "month": m,
            "profit_id": gold_profit[idx]["profit_id"],
            "flight_info_id": gold_flight_info[idx]["flight_info_id"],
            "delay_id": gold_delay[idx]["delay_id"],
            "metrics_id": gold_most_pop[idx]["metrics_id"]
        })
df_gf=pd.DataFrame(gold_fact)
df_gf.to_csv("GOLD/gold_fact_monthly_performance_report.csv", index=False)

# 5.6 gold_monthly_performance.csv (flatten)
gm = (
    df_gf
    .merge(df_gp, on="profit_id")
    .merge(df_gfi, on="flight_info_id")
    .merge(df_gd, on="delay_id")
    .merge(df_gmp, on="metrics_id")
)
out=gm[[
    "year","month","total_flights","total_profit","average_delay",
    "most_popular_destination","most_popular_order_method"
]]
out.to_csv("GOLD/gold_monthly_performance.csv", index=False)

print("All CSVs generated successfully (with ticket_class in silver_tickets).")
