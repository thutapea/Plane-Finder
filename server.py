from flask import Flask, send_from_directory
from flask_sock import Sock
import json
import base64
import os
import folium
from ultralytics import YOLO
from angle_finder import coords, querry 

dist_threshold = 100
angle_threshold = 80.0
app = Flask(__name__)
sock = Sock(app)

model = YOLO("best.pt")
print(model.names)
def get_generic_label(class_id):
    #if class_id == 36: return "Bird"
    #if class_id == 37: return "Helicopter"
    #if class_id == 38: return "Drone"
    return "p-airplane"  # Catch-all for everything else

def detect_plane(base64_data):
    #model = YOLO("best.pt")
    img_data = base64.b64decode(base64_data)
    with open("temp_image.jpg", "wb") as f: 
        f.write(img_data)
    os.rename("temp_image.jpg", "image.jpg")
    #with open("image.jpg", "wb") as f: 
    #    f.write(img_data)
    results = model.predict("image.jpg", conf=0.3, verbose=False) 
        
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = get_generic_label(int(class_id))
            print(f"DEBUG: Detected class '{class_name}' with confidence {box.conf[0]:.2f}")
            if class_name == "p-airplane":                    
                return True                    
    return False

def update_folium_map(lat, lon, plane):
    m = folium.Map(location=[lat, lon], zoom_start=12)
    folium.Marker([lat, lon], icon=folium.Icon()).add_to(m)
    folium.Marker([plane['lat'], plane['lon']], popup=plane['callsign'], icon=folium.Icon(icon="plane")).add_to(m)
    m.save("flight_map.html")

@sock.route('/')
def websocket_handler(ws):
    while True:
        message = ws.receive()
        if not message:
            break

        data = json.loads(message)
        telemetry_data = data.get("telemetry", {})
        #print(f"messagecont: {message}")
        lat = telemetry_data.get("lat")
        lon = telemetry_data.get("lon")
        alt = telemetry_data.get("alt")
        heading = telemetry_data.get("heading")
        pitch = telemetry_data.get("pitch")
        base64_image = data.get("image")
        print(f"Data: Lat={lat}, Lon={lon}, Pitch={pitch}")
        if lat is None or lon is None:
            print("Waiting for GPS lat/lon is None)")
            continue

        print("Received image")

        if detect_plane(base64_image):
            print("Plane detected")
            print(f"DEBUG: lat={lat}, lon={lon}, alt={alt}")
            location_A = coords(lat, lon, alt if alt is not None else 0)
            plane_targets = querry(location_A)

            best_match = None
            min_dist_sq = float("inf")

            for plane in plane_targets:
                target_bearing = plane['target_bearing']
                target_elevation = plane['target_elevation']
                plane_dist_sq = plane['dist']

                bearing_diff = abs(heading - target_bearing)
                if bearing_diff > 180:
                    bearing_diff = 360 - bearing_diff
                pitch_diff = abs(pitch - target_elevation)

                if bearing_diff < angle_threshold and pitch_diff < angle_threshold:
                    if plane_dist_sq < dist_threshold and plane_dist_sq < min_dist_sq:
                        min_dist_sq = plane_dist_sq
                        best_match = plane
                        best_match['bearing_diff'] = bearing_diff
                        best_match['pitch_diff'] = pitch_diff

            if best_match:
                print(f"Plane is {best_match['callsign']}")
                print(f"Angle: heading={best_match['target_bearing']}, pitch={best_match['target_elevation']}")
                print(f"Diffs: heading={best_match['bearing_diff']}, pitch={best_match['pitch_diff']}")
                update_folium_map(lat, lon, best_match)

@app.route("/map")
def serve_map():
    if not os.path.exists("flight_map.html"):
        m = folium.Map(location=[0, 0], zoom_start=2)
        m.save("flight_map.html")
    return send_from_directory(".", "flight_map.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)