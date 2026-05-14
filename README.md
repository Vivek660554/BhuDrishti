# 🛰️ BhuDrishti
### AI-Powered Spatial Asset Management System

> Built for the **eGov Foundation × Indian Railways Hackathon**
> Detecting and classifying urban assets from satellite and drone imagery — powering smarter land governance for Indian Railways.

---

## 📌 Problem Statement

Indian Railways manages over **68,000 route kilometres** of land, stations, bridges, tracks, drainage systems, and green cover — most of it tracked via paper registers and siloed spreadsheets. This leads to:

- Illegal encroachments going undetected for years
- Revenue leakage from unmapped properties
- Poor maintenance of critical drainage and sewage systems
- Green cover loss without early warning
- Emergency response hampered by inaccurate asset data

**BhuDrishti** solves this by using AI and satellite imagery to automatically detect, classify, and map these assets in real time.

---

## 🎯 What It Does

BhuDrishti is an end-to-end AI solution that:

- **Ingests** aerial, satellite, or drone imagery of railway land and urban areas
- **Detects and classifies** urban assets automatically using a computer vision model
- **Overlays** detection results on the uploaded image with labeled bounding boxes and confidence scores
- **Outputs metadata** — asset type, estimated area (sq. meters), and geo-coordinates
- **Exports** results as GeoJSON / CSV for downstream use in DIGIT or GIS tools

---

## 🗂️ Asset Categories Detected

| Category               | Examples                              | Status        |
|------------------------|---------------------------------------|---------------|
| Properties & Buildings | Residential, commercial, rooftop area |  Must Have    |
| Trees & Green Cover    | Individual trees, canopy clusters     |  Must Have    |
| Parks & Open Spaces    | Playgrounds, gardens, open plots      |  Must Have    |
| Water Bodies           | Lakes, ponds, rivers, canals          |  Must Have    |
| Roads & Footpaths      | Road network, pedestrian paths        |  Must Have    |
| Drains & Sewage        | Open drains, stormwater channels      |  Must Have    |
| Vehicles & Parking     | Parked vehicles, parking lots         |  Good to Have |
| Waste Dumps            | Illegal dumping sites, landfills      |  Good to Have |
| Solar Panels           | Rooftop solar installations           |  Bonus        |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                     BhuDrishti                      │
│                                                     │
│  ┌──────────┐     ┌──────────┐    ┌──────────────┐  │
│  │ Frontend │───▶│  FastAPI │───▶│  YOLOv8 /    │  │
│  │ React.js │     │ Backend  │    │  SAM Model   │  │
│  └──────────┘     └──────────┘    └──────────────┘  │
│       │               │                │            │
│       ▼               ▼                ▼            │
│  Leaflet.js      GeoJSON /        DeepGlobe /       │
│  Map Overlay     CSV Export       SpaceNet Data     │
└─────────────────────────────────────────────────────┘
```

**Flow:**
1. User uploads satellite / drone image via the React UI
2. Image sent to FastAPI `/detect` endpoint
3. YOLOv8 model runs inference and returns detections
4. Frontend overlays bounding boxes with labels and confidence scores
5. Summary panel shows count + estimated area per category
6. User can export results as GeoJSON for DIGIT / QGIS

---

## 🛠️ Tech Stack

| Layer             | Technology                             |
|-------------------|----------------------------------------|
| ML / CV Model     | YOLOv8 (Ultralytics), SAM (Meta)       |
| Backend           | Python, FastAPI                        |
| Frontend          | React.js, Leaflet.js                   |
| GIS Processing    | Geopandas, Shapely, GDAL               |
| Image Processing  | OpenCV, PIL                            |
| Training Data     | DeepGlobe, SpaceNet, iSAID, INRIA      |
| Satellite Imagery | Bhuvan ISRO, Sentinel-2, OpenAerialMap |
| GIS Data          | OpenStreetMap, GeoFabrik               |
| Export Formats    | GeoJSON, Shapefile, CSV                |
| Training Infra    | Google Colab (GPU)                     |

---

## 📁 Project Structure

```
bhu-drishti/
├── backend/
│   ├── main.py               # FastAPI app entry point
│   ├── detect.py             # YOLOv8 inference logic
│   ├── geo_utils.py          # Area calculation, geo-coordinate processing
│   ├── export.py             # GeoJSON / CSV export
│   └── requirements.txt
│
├── model/
│   ├── train.py              # Fine-tuning script on DeepGlobe
│   ├── weights/              # Trained model weights (.pt files)
│   └── dataset.yaml          # Dataset config for YOLOv8
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── ImageUpload.jsx
│   │   │   ├── DetectionOverlay.jsx
│   │   │   ├── SummaryPanel.jsx
│   │   │   └── MapView.jsx
│   │   └── index.jsx
│   ├── public/
│   └── package.json
│
├── data/
│   └── sample_images/        # Test satellite / drone images
│
├── notebooks/
│   └── training.ipynb        # Colab training notebook
│
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- GPU recommended for inference (CPU works for demo)

### 1. Clone the repository

```bash
git clone https://github.com/your-team/bhu-drishti.git
cd bhu-drishti
```

### 2. Set up the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Set up the frontend

```bash
cd frontend
npm install
npm start
```

### 4. Open the app

Visit `http://localhost:3000` — upload a satellite image and see detections instantly.

---

## 🔌 API Reference

### `POST /detect`

Accepts an image file and returns detected assets.

**Request:**
```
Content-Type: multipart/form-data
Body: image (file)
```

**Response:**
```json
{
  "detections": [
    {
      "asset_type": "water_body",
      "confidence": 0.91,
      "bbox": [120, 340, 280, 480],
      "area_sqm": 4200.5,
      "geo_coords": { "lat": 19.876, "lng": 75.342 }
    }
  ],
  "summary": {
    "water_body": { "count": 2, "total_area_sqm": 7400 },
    "buildings": { "count": 14, "total_area_sqm": 32000 }
  }
}
```

### `GET /export`

Returns the latest detection results as a downloadable GeoJSON file.

---

## 🗺️ Data Sources

| Source | Used For | Link |
|--------------------|----------------------------------------|-------------------------------------------------------------------------|
| DeepGlobe (Kaggle) | Primary training dataset               | [kaggle.com](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset) |
| SpaceNet           | Buildings & roads training             | [spacenet.ai](https://spacenet.ai/datasets)                             |
| iSAID              | Instance segmentation in aerial images | [captain-whu.github.io](https://captain-whu.github.io/iSAID/index.html) |
| Bhuvan ISRO        | India satellite imagery                | [bhuvan.nrsc.gov.in](https://bhuvan.nrsc.gov.in)                        |
| Sentinel-2         | Free satellite imagery (ESA)           | [Copernicus Hub](https://scihub.copernicus.eu)                          |
| OpenAerialMap      | Open drone imagery                     | [openaerialmap.org](https://openaerialmap.org)                          |
| OpenStreetMap      | GIS base layers                        | [geofabrik.de](https://download.geofabrik.de)                           |

---

## ✨ Features

- **Image Upload & Inference UI** — upload any aerial or satellite image and see results in seconds
- **Asset Detection Overlay** — colored bounding boxes with labels and confidence scores per asset type
- **Area & Count Summary** — per-category totals with estimated sq. meter areas
- **GIS Map View** — detected assets overlaid on a Leaflet.js base map with geo-coordinates
- **GeoJSON Export** — download results for use in QGIS, DIGIT, or any GIS tool
- **Change Detection** — compare two images of the same area to detect encroachments, tree felling, and new construction

---

## 🔮 Bonus Features

- **DIGIT Integration** — push detected asset data to a mock DIGIT Urban Asset Registry endpoint
- **3D Height Estimation** — estimate building height from shadow analysis
- **Real-time Stream Processing** — process live drone video feed for real-time detection

---

## 👥 Team

| Name                   | Role                                                    |
|------------------------|---------------------------------------------------------|
| [Vivek Jayadev]        | Team Leader — ML Engineer, Backend & GIS Processing     |
| [Ojas Damodhar Dhenge] | Team Manager — Solution Architecture & Coordination     |
| [Aryan]                | Frontend Developer — React UI & Map Integration         |

---

## 🙏 Acknowledgements

- [eGov Foundation](https://egov.org.in) & [DIGIT Platform](https://urban.digit.org) for the problem statement
- [Ultralytics YOLOv8](https://docs.ultralytics.com) for the detection framework
- [Meta AI — Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- [Indian Space Research Organisation (ISRO) — Bhuvan](https://bhuvan.nrsc.gov.in)
- DeepGlobe, SpaceNet, and iSAID dataset contributors

---