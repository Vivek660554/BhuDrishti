import io
import json
import numpy as np
import rasterio
from rasterio.io import MemoryFile  # <-- Added for in-memory file handling
from pyproj import Transformer
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image

app = FastAPI(title="Satellite Asset Detection API")

model = YOLO("yolov8n.pt")

detection_history = []

def get_geo_coords(x, y, dataset):
    """Returns (lat, lon) in WGS84 coordinates."""
    if dataset.transform:
        # 1. Get coordinates in the dataset's native CRS
        native_x, native_y = dataset.xy(int(y), int(x))
        
        # 2. Convert native CRS to EPSG:4326 (Lat/Lon) if necessary
        if dataset.crs and dataset.crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(native_x, native_y)
        else:
            lon, lat = native_x, native_y
            
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
    
    # Standard image parsing for YOLO
    img = Image.open(io.BytesIO(contents))
    results = model.predict(img)
    
    detections = []
    
    # Correctly open rasterio using MemoryFile and keep it open during processing
    with MemoryFile(contents) as memfile:
        try:
            ds = memfile.open()
        except Exception:
            ds = None

        try:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0].tolist()
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
                        "geo_coords": {"lat": lat, "lon": lon} if lat is not None else "No GeoData"
                    }
                    detections.append(det_data)
                    detection_history.append(det_data)
        finally:
            if ds:
                ds.close()

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
