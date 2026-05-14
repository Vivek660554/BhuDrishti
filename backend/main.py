import io
import json
import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from pyproj import Transformer
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image

app = FastAPI(title="Satellite Asset Detection API")

model = YOLO("best.pt")

detection_history = []

def get_geo_coords(x, y, dataset):
    if dataset.transform:
        lon, lat = dataset.xy(y, x)
        return lat, lon
    return None, None

def calculate_area(bbox, dataset):
    x1, y1, x2, y2 = bbox
    pixel_width = abs(dataset.res[0])
    pixel_height = abs(dataset.res[1])
    
    width_m = (x2 - x1) * pixel_width
    height_m = (y2 - y1) * pixel_height
    return round(width_m * height_m, 2)

@app.post("/detect")
async def detect_assets(file: UploadFile = File(...)):
    global detection_history
    contents = await file.read()
    
    img = Image.open(io.BytesIO(contents))
    results = model.predict(img)
    
    try:
        with rasterio.open(io.BytesIO(contents)) as ds:
            crs = ds.crs
            transform = ds.transform
    except Exception:
        ds = None

    detections = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            conf = float(box.conf)
            cls = int(box.cls)
            label = model.names[cls]
            
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            lat, lon = (None, None)
            area = 0
            
            if ds:
                lat, lon = get_geo_coords(cx, cy, ds)
                area = calculate_area(b, ds)

            det_data = {
                "asset_type": label,
                "confidence": conf,
                "bbox": b,
                "area_sqm": area,
                "geo_coords": {"lat": lat, "lon": lon} if lat else "No GeoData"
            }
            detections.append(det_data)
            detection_history.append(det_data)

    return {"filename": file.filename, "detections": detections}

@app.post("/export")
async def export_geojson():
    features = []
    for det in detection_history:
        if det["geo_coords"] != "No GeoData":
            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [det["geo_coords"]["lon"], det["geo_coords"]["lat"]]
                },
                "properties": {
                    "class": det["asset_type"],
                    "confidence": det["confidence"],
                    "area": det["area_sqm"]
                }
            }
            features.append(feat)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    return JSONResponse(content=geojson)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
