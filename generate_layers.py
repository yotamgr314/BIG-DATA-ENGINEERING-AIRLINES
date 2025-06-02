#!/usr/bin/env python3
"""
This script generates synthetic CSV files for Bronze, Silver, and Gold layers.
Each layer has its own subfolder, and each CSV file is created with sample data
matching the specified schema. If the folders or files do not exist, they are created.
"""

import os
import uuid
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# ---------------------
# Configuration
# ---------------------

# Number of sample rows per file (adjust as needed)
NUM_ROWS_BRONZE = 100
NUM_ROWS_SILVER = 100
NUM_ROWS_GOLD = 50

# Define folder names
FOLDERS = {
    "bronze": [
        "flight_routs.csv",
        "flights.csv",
        "weather.csv",
        "tickets_prices.csv",
        "booked_ticket.csv",
        "flights_reports.csv",
    ],
    "silver": [
        "silver_flight_routs.csv",
        "silver_flights.csv",
        "silver_tickets_prices.csv",
        "silver_weather.csv",
        "silver_tickets.csv",
    ],
    "gold": [
        "gold_monthly_performance.csv",
        "gold_ml_training.csv",
    ],
}

# ---------------------
# Helper functions
# ---------------------

def ensure_folder(path: str):
    """Create a folder if it doesn't already exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def random_airport():
    """Return a random 3-letter airport code."""
    return "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=3))

def random_datetime(start: datetime, end: datetime) -> datetime:
    """Return a random datetime between start and end."""
    delta = end - start
    int_delta = int(delta.total_seconds())
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)

def random_condition():
    """Pick a random weather condition."""
    return random.choice(["Clear", "Cloudy", "Rain", "Snow", "Fog", "Storm"])

def random_order_method():
    """Pick a random order method."""
    return random.choice(["Online", "Agent", "Kiosk"])

def random_name():
    """Generate a random first or last name."""
    names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis",
        "Garcia", "Rodriguez", "Wilson", "Martinez", "Anderson", "Taylor",
        "Thomas", "Hernandez", "Moore", "Martin", "Jackson", "Thompson",
    ]
    return random.choice(names)

# ---------------------
# Generate Bronze Layer
# ---------------------

def generate_bronze():
    base_dir = "bronze"
    ensure_folder(base_dir)

    # We'll need these references for dependent rows
    now = datetime.utcnow()
    base_date = now.date()

    # 1. flight_routs.csv
    routes = {
        "airport_origin": [],
        "airport_destination": [],
        "latitude_origin": [],
        "longitude_origin": [],
        "latitude_destination": [],
        "longitude_destination": [],
        "time_of_flight": [],
        "distance": [],
    }
    for _ in range(NUM_ROWS_BRONZE):
        origin = random_airport()
        dest = random_airport()
        while dest == origin:
            dest = random_airport()
        lat_o, lon_o = round(random.uniform(-90, 90), 6), round(random.uniform(-180, 180), 6)
        lat_d, lon_d = round(random.uniform(-90, 90), 6), round(random.uniform(-180, 180), 6)
        flight_time = round(random.uniform(0.5, 15.0), 2)  # hours
        distance = round(flight_time * random.uniform(400, 550), 2)  # approximate km

        routes["airport_origin"].append(origin)
        routes["airport_destination"].append(dest)
        routes["latitude_origin"].append(lat_o)
        routes["longitude_origin"].append(lon_o)
        routes["latitude_destination"].append(lat_d)
        routes["longitude_destination"].append(lon_d)
        routes["time_of_flight"].append(flight_time)
        routes["distance"].append(distance)

    df_routes = pd.DataFrame(routes)
    df_routes.to_csv(os.path.join(base_dir, "flight_routs.csv"), index=False)

    # 2. flights.csv (streaming data)
    flights = {
        "flight_id": [],
        "departure_time": [],
        "arrival_time": [],
        "airport_origin": [],
        "airport_destination": [],
        "average_height": [],
        "status": [],
        "status_change_time": [],
        "actual": [],
    }
    for _ in range(NUM_ROWS_BRONZE):
        fid = str(uuid.uuid4())
        origin = random.choice(df_routes["airport_origin"])
        dest = random.choice(df_routes["airport_destination"])
        dep = random_datetime(now - timedelta(days=30), now)
        dur_hours = float(df_routes.loc[df_routes["airport_origin"] == origin, "time_of_flight"].sample().iloc[0])
        arr = dep + timedelta(hours=dur_hours)
        avg_height = round(random.uniform(3000, 12000), 2)  # in meters
        status = random.choice(["Scheduled", "En Route", "Landed", "Cancelled"])
        status_time = dep + timedelta(minutes=random.randint(0, int((arr - dep).total_seconds() / 60)))
        actual_flag = random.choice([True, False])

        flights["flight_id"].append(fid)
        flights["departure_time"].append(dep)
        flights["arrival_time"].append(arr)
        flights["airport_origin"].append(origin)
        flights["airport_destination"].append(dest)
        flights["average_height"].append(avg_height)
        flights["status"].append(status)
        flights["status_change_time"].append(status_time)
        flights["actual"].append(actual_flag)

    df_flights = pd.DataFrame(flights)
    df_flights.to_csv(os.path.join(base_dir, "flights.csv"), index=False)

    # 3. weather.csv (API data)
    weather = {
        "flight_id": [],
        "point_index": [],
        "latitude": [],
        "longitude": [],
        "height": [],
        "temperature": [],
        "humidity": [],
        "wind_speed": [],
        "wind_direction": [],
        "condition": [],
        "sample_for_date": [],
        "time_stamp": [],
    }
    for _ in range(NUM_ROWS_BRONZE):
        fid = random.choice(df_flights["flight_id"])
        idx = random.randint(0, 10)
        lat = round(random.uniform(-90, 90), 6)
        lon = round(random.uniform(-180, 180), 6)
        height = round(random.uniform(0, 12000), 2)
        temp = round(random.uniform(-50, 40), 2)  # Celsius
        hum = round(random.uniform(0, 100), 2)  # %
        wind_spd = round(random.uniform(0, 150), 2)  # km/h
        wind_dir = round(random.uniform(0, 360), 2)  # degrees
        cond = random_condition()
        sample_date = random_datetime(now - timedelta(days=30), now).date()
        time_stmp = random_datetime(
            datetime.combine(sample_date, datetime.min.time()),
            datetime.combine(sample_date, datetime.max.time()),
        )

        weather["flight_id"].append(fid)
        weather["point_index"].append(idx)
        weather["latitude"].append(lat)
        weather["longitude"].append(lon)
        weather["height"].append(height)
        weather["temperature"].append(temp)
        weather["humidity"].append(hum)
        weather["wind_speed"].append(wind_spd)
        weather["wind_direction"].append(wind_dir)
        weather["condition"].append(cond)
        weather["sample_for_date"].append(sample_date)
        weather["time_stamp"].append(time_stmp)

    df_weather = pd.DataFrame(weather)
    df_weather.to_csv(os.path.join(base_dir, "weather.csv"), index=False)

    # 4. tickets_prices.csv (SCD Type 2)
    tp = {
        "price_id": [],
        "airport_origin": [],
        "airport_destination": [],
        "first_class_price": [],
        "business_class_price": [],
        "economy_class_price": [],
        "first_class_luggage": [],
        "business_class_luggage": [],
        "economy_class_luggage": [],
        "start_date": [],
        "end_date": [],
        "actual": [],
    }
    for _ in range(NUM_ROWS_BRONZE):
        pid = str(uuid.uuid4())
        origin = random_airport()
        dest = random_airport()
        while dest == origin:
            dest = random_airport()
        fc_price = round(random.uniform(2000, 5000), 2)
        bc_price = round(random.uniform(1000, 2000), 2)
        ec_price = round(random.uniform(200, 1000), 2)
        fc_lug = round(random.uniform(50, 100), 2)  # kilos
        bc_lug = round(random.uniform(40, 80), 2)
        ec_lug = round(random.uniform(20, 50), 2)
        start = base_date - timedelta(days=random.randint(0, 365))
        end = start + timedelta(days=random.randint(30, 180))
        actual_flag = random.choice([True, False])

        tp["price_id"].append(pid)
        tp["airport_origin"].append(origin)
        tp["airport_destination"].append(dest)
        tp["first_class_price"].append(fc_price)
        tp["business_class_price"].append(bc_price)
        tp["economy_class_price"].append(ec_price)
        tp["first_class_luggage"].append(fc_lug)
        tp["business_class_luggage"].append(bc_lug)
        tp["economy_class_luggage"].append(ec_lug)
        tp["start_date"].append(start)
        tp["end_date"].append(end)
        tp["actual"].append(actual_flag)

    df_tp = pd.DataFrame(tp)
    df_tp.to_csv(os.path.join(base_dir, "tickets_prices.csv"), index=False)

    # 5. booked_ticket.csv
    bt = {
        "ticket_id": [],
        "flight_id": [],
        "ticket_class": [],
        "luggage_class": [],
        "passenger_id": [],
        "passenger_first_name": [],
        "passenger_second_name": [],
        "order_method": [],
        "booking_date": [],
    }
    for _ in range(NUM_ROWS_BRONZE):
        tid = str(uuid.uuid4())
        fid = random.choice(df_flights["flight_id"])
        tclass = random.choice(["First", "Business", "Economy"])
        lugclass = random.choice(["First", "Business", "Economy"])
        pid = str(uuid.uuid4())
        fname = random_name()
        sname = random_name()
        order = random_order_method()
        bdate = random_datetime(now - timedelta(days=365), now).date()

        bt["ticket_id"].append(tid)
        bt["flight_id"].append(fid)
        bt["ticket_class"].append(tclass)
        bt["luggage_class"].append(lugclass)
        bt["passenger_id"].append(pid)
        bt["passenger_first_name"].append(fname)
        bt["passenger_second_name"].append(sname)
        bt["order_method"].append(order)
        bt["booking_date"].append(bdate)

    df_bt = pd.DataFrame(bt)
    df_bt.to_csv(os.path.join(base_dir, "booked_ticket.csv"), index=False)

    # 6. flights_reports.csv
    fr = {
        "flight_id": [],
        "actual_departure_time": [],
        "actual_arrival_time": [],
        "delay_in_minutes": [],
        "report_date": [],
    }
    for _ in range(NUM_ROWS_BRONZE):
        fid = random.choice(df_flights["flight_id"])
        dep = random_datetime(now - timedelta(days=30), now)
        arr = dep + timedelta(hours=random.uniform(0.5, 15.0))
        delay = round(random.uniform(0, 300), 2)  # minutes
        rdate = dep.date()

        fr["flight_id"].append(fid)
        fr["actual_departure_time"].append(dep)
        fr["actual_arrival_time"].append(arr)
        fr["delay_in_minutes"].append(delay)
        fr["report_date"].append(rdate)

    df_fr = pd.DataFrame(fr)
    df_fr.to_csv(os.path.join(base_dir, "flights_reports.csv"), index=False)


# ---------------------
# Generate Silver Layer
# ---------------------

def generate_silver():
    base_dir = "silver"
    ensure_folder(base_dir)

    # Define now and base_date for this scope
    now = datetime.utcnow()
    base_date = now.date()

    # 1. silver_flight_routs.csv
    sr = {
        "airport_origin": [],
        "airport_destination": [],
        "latitude_origin": [],
        "longitude_origin": [],
        "latitude_destination": [],
        "longitude_destination": [],
        "time_of_flight": [],
        "distance": [],
    }
    for _ in range(NUM_ROWS_SILVER):
        origin = random_airport()
        dest = random_airport()
        while dest == origin:
            dest = random_airport()
        lat_o, lon_o = round(random.uniform(-90, 90), 6), round(random.uniform(-180, 180), 6)
        lat_d, lon_d = round(random.uniform(-90, 90), 6), round(random.uniform(-180, 180), 6)
        flight_time = round(random.uniform(0.5, 15.0), 2)
        distance = round(flight_time * random.uniform(400, 550), 2)

        sr["airport_origin"].append(origin)
        sr["airport_destination"].append(dest)
        sr["latitude_origin"].append(lat_o)
        sr["longitude_origin"].append(lon_o)
        sr["latitude_destination"].append(lat_d)
        sr["longitude_destination"].append(lon_d)
        sr["time_of_flight"].append(flight_time)
        sr["distance"].append(distance)

    df_sr = pd.DataFrame(sr)
    df_sr.to_csv(os.path.join(base_dir, "silver_flight_routs.csv"), index=False)

    # 2. silver_flights.csv
    sf = {
        "flight_id": [],
        "airport_origin": [],
        "airport_destination": [],
        "estimated_departure_time": [],
        "estimated_arrival_time": [],
        "actual_departure_time": [],
        "actual_arrival_time": [],
        "delay_in_minutes": [],
        "report_date": [],
        "number_of_passengers": [],
        "number_of_luggage": [],
        "total_flight_profit": [],
        "status": [],
        "status_change_time": [],
        "actual": [],
    }
    for _ in range(NUM_ROWS_SILVER):
        fid = str(uuid.uuid4())
        origin = random.choice(df_sr["airport_origin"])
        dest = random.choice(df_sr["airport_destination"])
        est_dep = random_datetime(now - timedelta(days=30), now)
        flight_hours = float(df_sr.loc[
            df_sr["airport_origin"] == origin, "time_of_flight"
        ].sample().iloc[0])
        est_arr = est_dep + timedelta(hours=flight_hours)
        act_dep = est_dep + timedelta(minutes=random.uniform(-30, 30))
        act_arr = est_arr + timedelta(minutes=random.uniform(-30, 30))
        delay = round((act_dep - est_dep).total_seconds() / 60.0, 2)
        rdate = est_dep.date()
        num_pass = random.randint(50, 300)
        num_lug = random.randint(50, 200)
        profit = round(random.uniform(10000, 500000), 2)
        status = random.choice(["Scheduled", "En Route", "Landed", "Delayed", "Cancelled"])
        status_time = est_dep + timedelta(minutes=random.randint(0, int((est_arr - est_dep).total_seconds() / 60)))
        actual_flag = random.choice([True, False])

        sf["flight_id"].append(fid)
        sf["airport_origin"].append(origin)
        sf["airport_destination"].append(dest)
        sf["estimated_departure_time"].append(est_dep)
        sf["estimated_arrival_time"].append(est_arr)
        sf["actual_departure_time"].append(act_dep)
        sf["actual_arrival_time"].append(act_arr)
        sf["delay_in_minutes"].append(delay)
        sf["report_date"].append(rdate)
        sf["number_of_passengers"].append(num_pass)
        sf["number_of_luggage"].append(num_lug)
        sf["total_flight_profit"].append(profit)
        sf["status"].append(status)
        sf["status_change_time"].append(status_time)
        sf["actual"].append(actual_flag)

    df_sf = pd.DataFrame(sf)
    df_sf.to_csv(os.path.join(base_dir, "silver_flights.csv"), index=False)

    # 3. silver_tickets_prices.csv
    stp = {
        "price_id": [],
        "airport_origin": [],
        "airport_destination": [],
        "first_class_price": [],
        "business_class_price": [],
        "economy_class_price": [],
        "first_class_luggage": [],
        "business_class_luggage": [],
        "economy_class_luggage": [],
        "start_date": [],
        "end_date": [],
        "actual": [],
    }
    for _ in range(NUM_ROWS_SILVER):
        pid = str(uuid.uuid4())
        origin = random_airport()
        dest = random_airport()
        while dest == origin:
            dest = random_airport()
        fc_price = round(random.uniform(2000, 5000), 2)
        bc_price = round(random.uniform(1000, 2000), 2)
        ec_price = round(random.uniform(200, 1000), 2)
        fc_lug = round(random.uniform(50, 100), 2)
        bc_lug = round(random.uniform(40, 80), 2)
        ec_lug = round(random.uniform(20, 50), 2)
        start = base_date - timedelta(days=random.randint(0, 365))
        end = start + timedelta(days=random.randint(30, 180))
        actual_flag = random.choice([True, False])

        stp["price_id"].append(pid)
        stp["airport_origin"].append(origin)
        stp["airport_destination"].append(dest)
        stp["first_class_price"].append(fc_price)
        stp["business_class_price"].append(bc_price)
        stp["economy_class_price"].append(ec_price)
        stp["first_class_luggage"].append(fc_lug)
        stp["business_class_luggage"].append(bc_lug)
        stp["economy_class_luggage"].append(ec_lug)
        stp["start_date"].append(start)
        stp["end_date"].append(end)
        stp["actual"].append(actual_flag)

    df_stp = pd.DataFrame(stp)
    df_stp.to_csv(os.path.join(base_dir, "silver_tickets_prices.csv"), index=False)

    # 4. silver_weather.csv (aggregated per flight)
    sw = {
        "flight_id": [],
        "average_height": [],
        "average_temperature": [],
        "average_humidity": [],
        "average_wind_speed": [],
        "average_wind_direction": [],
        "most_common_condition": [],
        "number_of_most_common_condition": [],
    }
    for _ in range(NUM_ROWS_SILVER):
        fid = str(uuid.uuid4())
        avg_h = round(random.uniform(3000, 12000), 2)
        avg_t = round(random.uniform(-50, 40), 2)
        avg_hum = round(random.uniform(0, 100), 2)
        avg_ws = round(random.uniform(0, 150), 2)
        avg_wd = round(random.uniform(0, 360), 2)
        common_cond = random_condition()
        num_most = random.randint(1, 20)

        sw["flight_id"].append(fid)
        sw["average_height"].append(avg_h)
        sw["average_temperature"].append(avg_t)
        sw["average_humidity"].append(avg_hum)
        sw["average_wind_speed"].append(avg_ws)
        sw["average_wind_direction"].append(avg_wd)
        sw["most_common_condition"].append(common_cond)
        sw["number_of_most_common_condition"].append(num_most)

    df_sw = pd.DataFrame(sw)
    df_sw.to_csv(os.path.join(base_dir, "silver_weather.csv"), index=False)

    # 5. silver_tickets.csv
    st = {
        "ticket_id": [],
        "flight_id": [],
        "passenger_id": [],
        "passenger_first_name": [],
        "passenger_second_name": [],
        "order_method": [],
        "ticket_price": [],
        "booking_date": [],
    }
    for _ in range(NUM_ROWS_SILVER):
        tid = str(uuid.uuid4())
        fid = random.choice(df_sf["flight_id"])
        pid = str(uuid.uuid4())
        fname = random_name()
        sname = random_name()
        order = random_order_method()
        price = round(random.uniform(100, 2000), 2)
        bdate = random_datetime(now - timedelta(days=365), now).date()

        st["ticket_id"].append(tid)
        st["flight_id"].append(fid)
        st["passenger_id"].append(pid)
        st["passenger_first_name"].append(fname)
        st["passenger_second_name"].append(sname)
        st["order_method"].append(order)
        st["ticket_price"].append(price)
        st["booking_date"].append(bdate)

    df_st = pd.DataFrame(st)
    df_st.to_csv(os.path.join(base_dir, "silver_tickets.csv"), index=False)


# ---------------------
# Generate Gold Layer
# ---------------------

def generate_gold():
    base_dir = "gold"
    ensure_folder(base_dir)

    # 1. gold_monthly_performance.csv
    gmp = {
        "year": [],
        "month": [],
        "total_flights": [],
        "total_profit": [],
        "total_passengers_number": [],
        "average_passengers_number_on_flight": [],
        "average_ticket_price": [],
        "average_delay": [],
        "minimal_delay": [],
        "maximal_delay": [],
        "most_popular_origin": [],
        "most_popular_destination": [],
        "most_popular_order_method": [],
    }
    current_year = datetime.utcnow().year
    for i in range(NUM_ROWS_GOLD):
        year = random.choice([current_year - 1, current_year])
        month = random.randint(1, 12)
        total_flights = random.randint(500, 5000)
        total_profit = round(random.uniform(1e6, 1e7), 2)
        total_passengers = random.randint(50000, 300000)
        avg_passengers = round(total_passengers / total_flights, 2)
        avg_ticket_price = round(random.uniform(200, 1000), 2)
        avg_delay = round(random.uniform(5, 60), 2)
        min_delay = round(random.uniform(0, 5), 2)
        max_delay = round(random.uniform(60, 300), 2)
        popular_orig = random_airport()
        popular_dest = random_airport()
        while popular_dest == popular_orig:
            popular_dest = random_airport()
        popular_order = random_order_method()

        gmp["year"].append(year)
        gmp["month"].append(month)
        gmp["total_flights"].append(total_flights)
        gmp["total_profit"].append(total_profit)
        gmp["total_passengers_number"].append(total_passengers)
        gmp["average_passengers_number_on_flight"].append(avg_passengers)
        gmp["average_ticket_price"].append(avg_ticket_price)
        gmp["average_delay"].append(avg_delay)
        gmp["minimal_delay"].append(min_delay)
        gmp["maximal_delay"].append(max_delay)
        gmp["most_popular_origin"].append(popular_orig)
        gmp["most_popular_destination"].append(popular_dest)
        gmp["most_popular_order_method"].append(popular_order)

    df_gmp = pd.DataFrame(gmp)
    df_gmp.to_csv(os.path.join(base_dir, "gold_monthly_performance.csv"), index=False)

    # 2. gold_ml_training.csv
    gml = {
        "airport_origin": [],
        "airport_destination": [],
        "number_of_passengers": [],
        "number_of_luggage": [],
        "average_height": [],
        "average_temperature": [],
        "average_humidity": [],
        "average_wind_speed": [],
        "average_wind_direction": [],
        "most_common_condition": [],
        "number_of_most_common_condition": [],
        "time_of_flight": [],
        "distance": [],
        "delay_in_minutes": [],
    }
    for _ in range(NUM_ROWS_GOLD):
        origin = random_airport()
        dest = random_airport()
        while dest == origin:
            dest = random_airport()
        num_pass = random.randint(50, 300)
        num_lug = random.randint(50, 200)
        avg_h = round(random.uniform(3000, 12000), 2)
        avg_t = round(random.uniform(-50, 40), 2)
        avg_hum = round(random.uniform(0, 100), 2)
        avg_ws = round(random.uniform(0, 150), 2)
        avg_wd = round(random.uniform(0, 360), 2)
        cond = random_condition()
        num_cond = random.randint(1, 20)
        tof = round(random.uniform(0.5, 15.0), 2)
        dist = round(tof * random.uniform(400, 550), 2)
        delay = round(random.uniform(0, 300), 2)

        gml["airport_origin"].append(origin)
        gml["airport_destination"].append(dest)
        gml["number_of_passengers"].append(num_pass)
        gml["number_of_luggage"].append(num_lug)
        gml["average_height"].append(avg_h)
        gml["average_temperature"].append(avg_t)
        gml["average_humidity"].append(avg_hum)
        gml["average_wind_speed"].append(avg_ws)
        gml["average_wind_direction"].append(avg_wd)
        gml["most_common_condition"].append(cond)
        gml["number_of_most_common_condition"].append(num_cond)
        gml["time_of_flight"].append(tof)
        gml["distance"].append(dist)
        gml["delay_in_minutes"].append(delay)

    df_gml = pd.DataFrame(gml)
    df_gml.to_csv(os.path.join(base_dir, "gold_ml_training.csv"), index=False)


# ---------------------
# Main Execution
# ---------------------

if __name__ == "__main__":
    print("Generating Bronze layer CSV files...")
    generate_bronze()
    print("Bronze layer complete.\nGenerating Silver layer CSV files...")
    generate_silver()
    print("Silver layer complete.\nGenerating Gold layer CSV files...")
    generate_gold()
    print("Gold layer complete.\nAll CSV files generated successfully.")
