#!/usr/bin/env python3
"""
This script generates synthetic CSV files for Bronze, Silver, and Gold layers.
Each layer has its own subfolder, and each CSV file is created with sample data
matching the specified schema. If the folders or files do not exist, they are created.

The Gold layer now produces a star-schema: Dimension tables and Fact tables.
"""

import os
import uuid
import random
from datetime import datetime, timedelta
import pandas as pd

# ---------------------
# Configuration
# ---------------------

NUM_ROWS_BRONZE = 100
NUM_ROWS_SILVER = 100
NUM_ROWS_GOLD = 50

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
        # Dimension tables
        "airport_dim.csv",
        "date_dim.csv",
        "ticket_class_dim.csv",
        "order_method_dim.csv",
        "flight_status_dim.csv",
        "price_scenario_dim.csv",
        # Fact tables
        "fact_flight_transaction.csv",
        "fact_flight_operational.csv",
        "fact_agg_flight_performance_od.csv",
        "fact_monthly_summary.csv",
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

    # 6. flights_reports.csv (late-arrival data)
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
# Generate Gold Layer (Star Schema)
# ---------------------

def generate_gold():
    base_dir = "gold"
    ensure_folder(base_dir)

    # --- Dimension Tables ---

    # 1. airport_dim.csv
    #    Derive unique airports from silver_flight_routs.csv if it exists, otherwise generate a set
    silver_routes_path = os.path.join("silver", "silver_flight_routs.csv")
    if os.path.exists(silver_routes_path):
        df_sr = pd.read_csv(silver_routes_path, dtype=str)
        unique_origins = df_sr["airport_origin"].unique().tolist()
        unique_dests = df_sr["airport_destination"].unique().tolist()
        all_airports = sorted(set(unique_origins + unique_dests))
    else:
        # fallback to a random list if silver not generated yet
        all_airports = sorted({random_airport() for _ in range(20)})

    airport_dim = {
        "airport_id": list(range(1, len(all_airports) + 1)),
        "airport_code": all_airports,
        "latitude": [round(random.uniform(-90, 90), 6) for _ in all_airports],
        "longitude": [round(random.uniform(-180, 180), 6) for _ in all_airports],
        # (Optional additional attributes like city, country can be added)
    }
    df_airport_dim = pd.DataFrame(airport_dim)
    df_airport_dim.to_csv(os.path.join(base_dir, "airport_dim.csv"), index=False)

    # 2. date_dim.csv
    #    Generate a calendar from one year ago up to today
    today = datetime.utcnow().date()
    start_date = today - timedelta(days=365)
    all_dates = pd.date_range(start=start_date, end=today, freq="D")
    date_dim = {
        "date_id": list(range(1, len(all_dates) + 1)),
        "full_date": all_dates.strftime("%Y-%m-%d").tolist(),
        "year": [d.year for d in all_dates],
        "quarter": [((d.month - 1) // 3) + 1 for d in all_dates],
        "month": [d.month for d in all_dates],
        "day": [d.day for d in all_dates],
        "day_of_week": [d.isoweekday() for d in all_dates],  # 1=Monday, 7=Sunday
        # (Optional flags like holiday can be added)
    }
    df_date_dim = pd.DataFrame(date_dim)
    df_date_dim.to_csv(os.path.join(base_dir, "date_dim.csv"), index=False)

    # 3. ticket_class_dim.csv
    ticket_classes = ["First", "Business", "Economy"]
    ticket_class_dim = {
        "ticket_class_id": list(range(1, len(ticket_classes) + 1)),
        "ticket_class": ticket_classes,
    }
    df_ticket_class_dim = pd.DataFrame(ticket_class_dim)
    df_ticket_class_dim.to_csv(os.path.join(base_dir, "ticket_class_dim.csv"), index=False)

    # 4. order_method_dim.csv
    order_methods = ["Online", "Agent", "Kiosk"]
    order_method_dim = {
        "order_method_id": list(range(1, len(order_methods) + 1)),
        "order_method": order_methods,
    }
    df_order_method_dim = pd.DataFrame(order_method_dim)
    df_order_method_dim.to_csv(os.path.join(base_dir, "order_method_dim.csv"), index=False)

    # 5. flight_status_dim.csv
    flight_statuses = ["Scheduled", "En Route", "Landed", "Delayed", "Cancelled"]
    flight_status_dim = {
        "status_id": list(range(1, len(flight_statuses) + 1)),
        "status": flight_statuses,
    }
    df_flight_status_dim = pd.DataFrame(flight_status_dim)
    df_flight_status_dim.to_csv(os.path.join(base_dir, "flight_status_dim.csv"), index=False)

    # 6. price_scenario_dim.csv
    #    Read silver_tickets_prices.csv if exists, otherwise generate random entries
    silver_prices_path = os.path.join("silver", "silver_tickets_prices.csv")
    price_rows = []
    if os.path.exists(silver_prices_path):
        df_silver_prices = pd.read_csv(silver_prices_path, parse_dates=["start_date", "end_date"])
        for idx, row in df_silver_prices.iterrows():
            price_rows.append({
                "price_id": row["price_id"],
                "airport_origin": row["airport_origin"],
                "airport_destination": row["airport_destination"],
                "first_class_price": row["first_class_price"],
                "business_class_price": row["business_class_price"],
                "economy_class_price": row["economy_class_price"],
                "first_class_luggage": row["first_class_luggage"],
                "business_class_luggage": row["business_class_luggage"],
                "economy_class_luggage": row["economy_class_luggage"],
                "start_date": row["start_date"].strftime("%Y-%m-%d"),
                "end_date": row["end_date"].strftime("%Y-%m-%d"),
                "actual": row["actual"],
            })
    else:
        # fallback: synthetic price scenarios
        for _ in range(NUM_ROWS_GOLD):
            pid = str(uuid.uuid4())
            origin = random.choice(all_airports)
            dest = random.choice(all_airports)
            while dest == origin:
                dest = random.choice(all_airports)
            fc_price = round(random.uniform(2000, 5000), 2)
            bc_price = round(random.uniform(1000, 2000), 2)
            ec_price = round(random.uniform(200, 1000), 2)
            fc_lug = round(random.uniform(50, 100), 2)
            bc_lug = round(random.uniform(40, 80), 2)
            ec_lug = round(random.uniform(20, 50), 2)
            start = start_date + timedelta(days=random.randint(0, 365))
            end = start + timedelta(days=random.randint(30, 180))
            actual_flag = random.choice([True, False])

            price_rows.append({
                "price_id": pid,
                "airport_origin": origin,
                "airport_destination": dest,
                "first_class_price": fc_price,
                "business_class_price": bc_price,
                "economy_class_price": ec_price,
                "first_class_luggage": fc_lug,
                "business_class_luggage": bc_lug,
                "economy_class_luggage": ec_lug,
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
                "actual": actual_flag,
            })

    df_price_scenario_dim = pd.DataFrame(price_rows)
    df_price_scenario_dim.to_csv(os.path.join(base_dir, "price_scenario_dim.csv"), index=False)

    # --- Fact Tables ---

    # 1. fact_flight_transaction.csv
    #    Origin: silver_tickets.csv; join price_id randomly from price_scenario_dim
    silver_tickets_path = os.path.join("silver", "silver_tickets.csv")
    if os.path.exists(silver_tickets_path):
        df_silver_tickets = pd.read_csv(silver_tickets_path, parse_dates=["booking_date"])
    else:
        # fallback: generate random ticket events
        df_silver_tickets = pd.DataFrame({
            "ticket_id": [str(uuid.uuid4()) for _ in range(NUM_ROWS_GOLD)],
            "flight_id": [random.choice(all_airports) for _ in range(NUM_ROWS_GOLD)],  # misuse airport codes as flight IDs
            "passenger_id": [str(uuid.uuid4()) for _ in range(NUM_ROWS_GOLD)],
            "passenger_first_name": [random_name() for _ in range(NUM_ROWS_GOLD)],
            "passenger_second_name": [random_name() for _ in range(NUM_ROWS_GOLD)],
            "order_method": [random_order_method() for _ in range(NUM_ROWS_GOLD)],
            "ticket_price": [round(random.uniform(100, 2000), 2) for _ in range(NUM_ROWS_GOLD)],
            "booking_date": [start_date + timedelta(days=random.randint(0, 365)) for _ in range(NUM_ROWS_GOLD)],
        })

    price_ids = df_price_scenario_dim["price_id"].tolist()
    ticket_fact_rows = []
    for _, row in df_silver_tickets.iterrows():
        ticket_fact_rows.append({
            "ticket_id": row["ticket_id"],
            "flight_id": row["flight_id"],
            "ticket_class": random.choice(ticket_classes := ["First", "Business", "Economy"]),
            "order_method": row["order_method"],
            "booking_date": row["booking_date"].strftime("%Y-%m-%d"),
            "ticket_price": row["ticket_price"],
            "price_id": random.choice(price_ids),
        })
    df_fact_flight_transaction = pd.DataFrame(ticket_fact_rows)
    df_fact_flight_transaction.to_csv(os.path.join(base_dir, "fact_flight_transaction.csv"), index=False)

    # 2. fact_flight_operational.csv
    #    Origin: silver_flights.csv + silver_weather.csv
    silver_flights_path = os.path.join("silver", "silver_flights.csv")
    silver_weather_path = os.path.join("silver", "silver_weather.csv")

    if os.path.exists(silver_flights_path):
        df_silver_flights = pd.read_csv(
            silver_flights_path,
            parse_dates=["estimated_departure_time", "estimated_arrival_time", "actual_departure_time", "actual_arrival_time", "status_change_time"]
        )
    else:
        # fallback: generate random flight operational events
        df_silver_flights = pd.DataFrame({
            "flight_id": [str(uuid.uuid4()) for _ in range(NUM_ROWS_GOLD)],
            "airport_origin": [random.choice(all_airports) for _ in range(NUM_ROWS_GOLD)],
            "airport_destination": [random.choice(all_airports) for _ in range(NUM_ROWS_GOLD)],
            "estimated_departure_time": [start_date + timedelta(days=random.randint(0, 365), hours=random.randint(0,23)) for _ in range(NUM_ROWS_GOLD)],
            "estimated_arrival_time": [start_date + timedelta(days=random.randint(0, 365), hours=random.randint(0,23)) for _ in range(NUM_ROWS_GOLD)],
            "actual_departure_time": [start_date + timedelta(days=random.randint(0, 365), hours=random.randint(0,23)) for _ in range(NUM_ROWS_GOLD)],
            "actual_arrival_time": [start_date + timedelta(days=random.randint(0, 365), hours=random.randint(0,23)) for _ in range(NUM_ROWS_GOLD)],
            "delay_in_minutes": [round(random.uniform(0, 300), 2) for _ in range(NUM_ROWS_GOLD)],
            "report_date": [(start_date + timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d") for _ in range(NUM_ROWS_GOLD)],
            "number_of_passengers": [random.randint(50, 300) for _ in range(NUM_ROWS_GOLD)],
            "number_of_luggage": [random.randint(50, 200) for _ in range(NUM_ROWS_GOLD)],
            "total_flight_profit": [round(random.uniform(10000, 500000), 2) for _ in range(NUM_ROWS_GOLD)],
            "status": [random.choice(flight_statuses := ["Scheduled", "En Route", "Landed", "Delayed", "Cancelled"]) for _ in range(NUM_ROWS_GOLD)],
            "status_change_time": [start_date + timedelta(days=random.randint(0, 365), hours=random.randint(0,23)) for _ in range(NUM_ROWS_GOLD)],
            "actual": [random.choice([True, False]) for _ in range(NUM_ROWS_GOLD)],
        })

    if os.path.exists(silver_weather_path):
        df_silver_weather = pd.read_csv(silver_weather_path)
    else:
        df_silver_weather = pd.DataFrame({
            "flight_id": [row["flight_id"] for _, row in df_silver_flights.iterrows()],
            "average_height": [round(random.uniform(3000, 12000), 2) for _ in range(len(df_silver_flights))],
            "average_temperature": [round(random.uniform(-50, 40), 2) for _ in range(len(df_silver_flights))],
            "average_humidity": [round(random.uniform(0, 100), 2) for _ in range(len(df_silver_flights))],
            "average_wind_speed": [round(random.uniform(0, 150), 2) for _ in range(len(df_silver_flights))],
            "average_wind_direction": [round(random.uniform(0, 360), 2) for _ in range(len(df_silver_flights))],
            "most_common_condition": [random_condition() for _ in range(len(df_silver_flights))],
            "number_of_most_common_condition": [random.randint(1, 20) for _ in range(len(df_silver_flights))],
        })

    # Join on flight_id (left join)
    df_flight_operational = pd.merge(
        df_silver_flights,
        df_silver_weather,
        on="flight_id",
        how="left"
    ).fillna({
        "average_height": 0,
        "average_temperature": 0,
        "average_humidity": 0,
        "average_wind_speed": 0,
        "average_wind_direction": 0,
        "most_common_condition": "Unknown",
        "number_of_most_common_condition": 0,
    })

    # Keep relevant columns
    df_fact_flight_operational = df_flight_operational[[
        "flight_id",
        "airport_origin",
        "airport_destination",
        "actual_departure_time",
        "actual_arrival_time",
        "delay_in_minutes",
        "report_date",
        "number_of_passengers",
        "number_of_luggage",
        "total_flight_profit",
        "status",
        "status_change_time",
        "actual",
        "average_height",
        "average_temperature",
        "average_humidity",
        "average_wind_speed",
        "average_wind_direction",
        "most_common_condition",
        "number_of_most_common_condition",
    ]]
    df_fact_flight_operational.to_csv(os.path.join(base_dir, "fact_flight_operational.csv"), index=False)

    # 3. fact_agg_flight_performance_od.csv
    #    One row per origin–destination with aggregated metrics
    agg_rows = []
    for _ in range(NUM_ROWS_GOLD):
        origin = random.choice(all_airports)
        dest = random.choice(all_airports)
        while dest == origin:
            dest = random.choice(all_airports)
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

        agg_rows.append({
            "airport_origin": origin,
            "airport_destination": dest,
            "number_of_passengers": num_pass,
            "number_of_luggage": num_lug,
            "average_height": avg_h,
            "average_temperature": avg_t,
            "average_humidity": avg_hum,
            "average_wind_speed": avg_ws,
            "average_wind_direction": avg_wd,
            "most_common_condition": cond,
            "number_of_most_common_condition": num_cond,
            "time_of_flight": tof,
            "distance": dist,
            "delay_in_minutes": delay,
        })

    df_fact_agg_od = pd.DataFrame(agg_rows)
    df_fact_agg_od.to_csv(os.path.join(base_dir, "fact_agg_flight_performance_od.csv"), index=False)

    # 4. fact_monthly_summary.csv
    #    Similar to the original gold_monthly_performance but with explicit name
    monthly_rows = []
    for _ in range(NUM_ROWS_GOLD):
        year = random.choice([today.year - 1, today.year])
        month = random.randint(1, 12)
        total_flights = random.randint(500, 5000)
        total_profit = round(random.uniform(1e6, 1e7), 2)
        total_passengers = random.randint(50000, 300000)
        avg_passengers = round(total_passengers / total_flights, 2)
        avg_ticket_price = round(random.uniform(200, 1000), 2)
        avg_delay = round(random.uniform(5, 60), 2)
        min_delay = round(random.uniform(0, 5), 2)
        max_delay = round(random.uniform(60, 300), 2)
        popular_orig = random.choice(all_airports)
        popular_dest = random.choice(all_airports)
        while popular_dest == popular_orig:
            popular_dest = random.choice(all_airports)
        popular_order = random.choice(order_methods)

        monthly_rows.append({
            "year": year,
            "month": month,
            "total_flights": total_flights,
            "total_profit": total_profit,
            "total_passengers_number": total_passengers,
            "average_passengers_number_on_flight": avg_passengers,
            "average_ticket_price": avg_ticket_price,
            "average_delay": avg_delay,
            "minimal_delay": min_delay,
            "maximal_delay": max_delay,
            "most_popular_origin": popular_orig,
            "most_popular_destination": popular_dest,
            "most_popular_order_method": popular_order,
        })

    df_fact_monthly = pd.DataFrame(monthly_rows)
    df_fact_monthly.to_csv(os.path.join(base_dir, "fact_monthly_summary.csv"), index=False)


# ---------------------
# Main Execution
# ---------------------

if __name__ == "__main__":
    print("Generating Bronze layer CSV files...")
    generate_bronze()
    print("Bronze layer complete.\nGenerating Silver layer CSV files...")
    generate_silver()
    print("Silver layer complete.\nGenerating Gold layer (star schema) CSV files...")
    generate_gold()
    print("Gold layer complete.\nAll CSV files generated successfully.")
