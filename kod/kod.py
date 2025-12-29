import os
import datetime as dt
import googlemaps

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    raise SystemExit("GOOGLE_MAPS_API_KEY ayarlı değil. Önce environment'a ekle.")

gmaps = googlemaps.Client(key=API_KEY)

origins = ["ODTÜ, Ankara"]
destinations = ["Kızılay, Ankara"]

resp = gmaps.distance_matrix(
    origins=origins,
    destinations=destinations,
    mode="driving",
    departure_time=dt.datetime.now(),  # trafik için
    units="metric",
    language="tr",
)

e = resp["rows"][0]["elements"][0]
print("Origin:", resp["origin_addresses"][0])
print("Destination:", resp["destination_addresses"][0])
print("Element status:", e["status"])

if e["status"] == "OK":
    print("Mesafe:", e["distance"]["text"], f"({e['distance']['value']} m)")
    print("Süre:", e["duration"]["text"], f"({e['duration']['value']} sn)")
    if "duration_in_traffic" in e:
        print("Trafikli süre:", e["duration_in_traffic"]["text"])
else:
    print("Detay:", e)
