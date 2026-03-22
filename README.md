# Research Center Quality Classification

A machine learning pipeline that classifies UK research centers into **Premium**, **Standard**, or **Basic** quality tiers based on their internal infrastructure and nearby healthcare access.

Built with K-Means clustering and deployed as a FastAPI web service with Docker support.

---

## Project Structure

```
├── EDA_and_Model.ipynb          # Full ML pipeline: EDA, feature selection, clustering
├── app.py                       # FastAPI API with interactive web UI
├── cluster_model.pkl            # Trained model bundle (KMeans + scaler + metadata)
├── research_centers.csv         # Original dataset (50 UK research centers)
├── research_centers_classified.csv  # Dataset with cluster and tier labels added
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container image definition
├── docker-compose.yaml          # One-command deployment
├── .env.draft                   # Environment variable template
├── .gitignore
└── .dockerignore
```

---

## Approach

### 1. Exploratory Data Analysis

- Checked for missing values and data types — dataset is clean with 50 rows and 10 columns
- Visualised distributions of facility counts, hospital/pharmacy access, diversity, and density
- Built a correlation heatmap revealing strong relationships between features (r = 0.80–0.90)
- Used boxplots and scatter plots to identify natural groupings in the data

### 2. Feature Selection

Selected 5 numeric quality indicators and dropped identifiers, city names, and coordinates:

| Feature | Rationale |
|---------|-----------|
| `internalFacilitiesCount` | Core measure of a center's own capacity |
| `hospitals_10km` | Nearby hospital access matters for clinical trials |
| `pharmacies_10km` | Pharmacy availability supports patient recruitment |
| `facilityDiversity_10km` | Variety of nearby facilities indicates a well-served area |
| `facilityDensity_10km` | Overall density of surrounding healthcare infrastructure |

Features were standardised using `StandardScaler` (mean=0, std=1) since K-Means relies on Euclidean distance and raw feature scales vary significantly.

### 3. Clustering

- Used the **Elbow Method** (inertia vs. k) and **Silhouette Analysis** to confirm k=3 as the optimal number of clusters
- Trained K-Means with `n_init=10` and `random_state=42` for reproducibility
- Achieved a **silhouette score of 0.5519**, indicating well-separated clusters
- Mapped clusters to tiers by ranking cluster centroids on mean feature values

### 4. Results

| Tier | Centers | Characteristics |
|------|---------|-----------------|
| **Premium** | 17 | Highest facility counts, best hospital/pharmacy access, top diversity and density scores |
| **Standard** | 17 | Mid-range across all features |
| **Basic** | 16 | Lowest infrastructure and healthcare access |

Bonus visualisations included:
- Geographic map plotting centers by lat/lon, coloured by tier
- Radar chart comparing average feature profiles across tiers

---

## API

### Running Locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

The web UI is available at `http://localhost:8000` with interactive tooltips and example pre-fill buttons.

### Running with Docker

```bash
docker compose up --build
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Interactive web UI for single predictions |
| `POST` | `/predict` | Classify a single research center |
| `POST` | `/predict/batch` | Classify multiple centers in one request |
| `GET` | `/health` | Health check |
| `GET` | `/model-info` | Model metadata (features, tier distribution, silhouette score) |
| `GET` | `/docs` | Auto-generated Swagger documentation |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "internalFacilitiesCount": 9,
    "hospitals_10km": 3,
    "pharmacies_10km": 2,
    "facilityDiversity_10km": 0.82,
    "facilityDensity_10km": 0.45
  }'
```

### Example Response

```json
{
  "predictedCategory": "Premium",
  "confidence_note": "Close to cluster centre — high confidence",
  "tier_icon": "★★★",
  "feature_comparison": { ... },
  "tips": [ ... ]
}
```

---

## Tech Stack

- **Python 3.10**
- **scikit-learn** — K-Means clustering, StandardScaler, silhouette scoring
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualisation
- **FastAPI / Pydantic** — API with input validation
- **joblib** — model serialisation
- **Docker** — containerised deployment

---

## Requirements

```
fastapi
uvicorn
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

Install with:

```bash
pip install -r requirements.txt
```
