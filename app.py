from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model artifacts at startup
try:
    kmeans, scaler, selected_features, cluster_to_tier = joblib.load("cluster_model.pkl")
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.error("cluster_model.pkl not found. Run the notebook first to train the model.")
    raise


# Tier styling for the UI
TIER_CONFIG = {
    "Premium": {"color": "#2ecc71", "icon": "★★★", "badge": "🟢"},
    "Standard": {"color": "#3498db", "icon": "★★☆", "badge": "🔵"},
    "Basic": {"color": "#e74c3c", "icon": "★☆☆", "badge": "🔴"},
    "Unknown": {"color": "#95a5a6", "icon": "?", "badge": "⚪"},
}

# Tier benchmark averages (from training data)
TIER_BENCHMARKS = {
    "Premium":  {"internalFacilitiesCount": 9.53, "hospitals_10km": 3.47,
                 "pharmacies_10km": 4.12, "facilityDiversity_10km": 0.85,
                 "facilityDensity_10km": 0.54},
    "Standard": {"internalFacilitiesCount": 4.94, "hospitals_10km": 1.53,
                 "pharmacies_10km": 2.06, "facilityDiversity_10km": 0.56,
                 "facilityDensity_10km": 0.29},
    "Basic":    {"internalFacilitiesCount": 2.31, "hospitals_10km": 0.50,
                 "pharmacies_10km": 0.44, "facilityDiversity_10km": 0.28,
                 "facilityDensity_10km": 0.13},
}

FEATURE_TOOLTIPS = {
    "internalFacilitiesCount": {
        "label": "Internal Facilities",
        "tooltip": "Number of internal facilities such as labs, testing units, and workstations. Higher values indicate better-equipped centers.",
        "range": "Typical range: 1–11",
        "unit": "facilities",
    },
    "hospitals_10km": {
        "label": "Hospitals (10km)",
        "tooltip": "Number of hospitals within a 10km radius. More nearby hospitals means better emergency and specialist access.",
        "range": "Typical range: 0–4",
        "unit": "hospitals",
    },
    "pharmacies_10km": {
        "label": "Pharmacies (10km)",
        "tooltip": "Number of pharmacies within a 10km radius. Important for medication availability during clinical trials.",
        "range": "Typical range: 0–5",
        "unit": "pharmacies",
    },
    "facilityDiversity_10km": {
        "label": "Facility Diversity",
        "tooltip": "Diversity index (0–1) measuring how varied the nearby healthcare facilities are. 1.0 = maximum diversity.",
        "range": "Range: 0.0–1.0",
        "unit": "index",
    },
    "facilityDensity_10km": {
        "label": "Facility Density",
        "tooltip": "Approximate density of healthcare facilities per area within 10km. Higher = more concentrated services.",
        "range": "Typical range: 0.05–0.70",
        "unit": "density score",
    },
}


# FastAPI app setup
app = FastAPI(
    title="Research Center Quality Classifier",
    description=(
        "## Classify research centers into quality tiers\n\n"
        "This API uses a **K-Means clustering model** trained on UK research center data "
        "to classify centers into **Premium**, **Standard**, or **Basic** quality tiers.\n\n"
        "### How it works\n"
        "1. Submit 5 quality metrics for a research center\n"
        "2. The model scales your input using the same StandardScaler from training\n"
        "3. K-Means predicts which cluster the center belongs to\n"
        "4. The cluster is mapped to a quality tier with detailed feedback\n\n"
        "### Quick start\n"
        "- Visit **[Interactive UI](/)** for a visual form-based interface\n"
        "- Use **POST /predict** for API integration\n"
        "- Use **GET /model-info** to inspect model details\n"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# Request/response schemas

class ResearchCenterInput(BaseModel):
    """Input features for research center quality prediction."""
    internalFacilitiesCount: float = Field(
        ..., ge=0,
        title="Internal Facilities Count",
        description="Number of internal facilities (labs, testing units, workstations). Typical range: 1–11.",
        json_schema_extra={"example": 9}
    )
    hospitals_10km: float = Field(
        ..., ge=0,
        title="Hospitals within 10km",
        description="Number of hospitals within a 10km radius. Typical range: 0–4.",
        json_schema_extra={"example": 3}
    )
    pharmacies_10km: float = Field(
        ..., ge=0,
        title="Pharmacies within 10km",
        description="Number of pharmacies within a 10km radius. Typical range: 0–5.",
        json_schema_extra={"example": 4}
    )
    facilityDiversity_10km: float = Field(
        ..., ge=0, le=1,
        title="Facility Diversity Index",
        description="Diversity index (0–1) representing variety of nearby healthcare facilities. 1.0 = maximum diversity.",
        json_schema_extra={"example": 0.82}
    )
    facilityDensity_10km: float = Field(
        ..., ge=0,
        title="Facility Density Score",
        description="Density of nearby healthcare facilities per area. Typical range: 0.05–0.70.",
        json_schema_extra={"example": 0.45}
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Premium center example",
                    "value": {
                        "internalFacilitiesCount": 10,
                        "hospitals_10km": 4,
                        "pharmacies_10km": 5,
                        "facilityDiversity_10km": 0.88,
                        "facilityDensity_10km": 0.55,
                    },
                },
                {
                    "summary": "Standard center example",
                    "value": {
                        "internalFacilitiesCount": 5,
                        "hospitals_10km": 2,
                        "pharmacies_10km": 2,
                        "facilityDiversity_10km": 0.55,
                        "facilityDensity_10km": 0.30,
                    },
                },
                {
                    "summary": "Basic center example",
                    "value": {
                        "internalFacilitiesCount": 2,
                        "hospitals_10km": 0,
                        "pharmacies_10km": 0,
                        "facilityDiversity_10km": 0.20,
                        "facilityDensity_10km": 0.10,
                    },
                },
            ]
        }
    }


class FeatureComparison(BaseModel):
    """Comparison of input value against the tier's average."""
    feature: str
    your_value: float
    tier_average: float
    status: str  # "above", "below", "at"


class PredictionResponse(BaseModel):
    """Detailed prediction response with tier, confidence context, and benchmarks."""
    predictedCluster: int = Field(description="Numeric cluster label (0, 1, or 2)")
    predictedCategory: str = Field(description="Quality tier: Premium, Standard, or Basic")
    tierIcon: str = Field(description="Visual tier indicator")
    confidence_note: str = Field(description="How clearly this center fits the tier")
    comparison: list[FeatureComparison] = Field(
        description="How your input compares to the tier's average values"
    )
    tip: str = Field(description="Actionable suggestion based on the prediction")


# Endpoints

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def homepage():
    """Interactive UI for classifying research centers."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Research Center Quality Classifier</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0; min-height: 100vh;
    display: flex; justify-content: center; align-items: flex-start;
    padding: 40px 20px;
  }
  .container { max-width: 720px; width: 100%; }
  h1 {
    text-align: center; font-size: 1.8rem; margin-bottom: 6px;
    background: linear-gradient(90deg, #2ecc71, #3498db);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .subtitle { text-align: center; color: #999; margin-bottom: 30px; font-size: 0.9rem; }
  .card {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 32px; margin-bottom: 24px;
    backdrop-filter: blur(10px);
  }
  .field-group { margin-bottom: 20px; position: relative; }
  .field-group label {
    display: flex; align-items: center; gap: 8px;
    font-weight: 600; margin-bottom: 6px; font-size: 0.95rem;
  }
  .tooltip-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 18px; height: 18px; border-radius: 50%;
    background: rgba(52,152,219,0.3); color: #3498db;
    font-size: 11px; font-weight: 700; cursor: help; flex-shrink: 0;
  }
  .tooltip-icon:hover + .tooltip-text { opacity: 1; transform: translateY(0); pointer-events: auto; }
  .tooltip-text {
    position: absolute; top: -8px; left: 220px; z-index: 10;
    background: #1a1a2e; border: 1px solid #3498db; border-radius: 8px;
    padding: 10px 14px; font-size: 0.82rem; color: #ccc;
    width: 280px; opacity: 0; transform: translateY(4px);
    transition: all 0.2s ease; pointer-events: none; line-height: 1.5;
  }
  .tooltip-text .range { color: #3498db; font-weight: 600; display: block; margin-top: 4px; }
  input[type="number"] {
    width: 100%; padding: 12px 16px; border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.15); background: rgba(255,255,255,0.05);
    color: #fff; font-size: 1rem; outline: none; transition: border-color 0.2s;
  }
  input[type="number"]:focus { border-color: #3498db; }
  input[type="number"]::placeholder { color: #666; }
  .btn-row { display: flex; gap: 12px; margin-top: 8px; }
  button {
    flex: 1; padding: 14px; border: none; border-radius: 10px;
    font-size: 1rem; font-weight: 600; cursor: pointer; transition: all 0.2s;
  }
  .btn-predict {
    background: linear-gradient(135deg, #2ecc71, #27ae60); color: #fff;
  }
  .btn-predict:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(46,204,113,0.3); }
  .btn-predict:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }
  .btn-example { background: rgba(52,152,219,0.2); color: #3498db; border: 1px solid rgba(52,152,219,0.3); }
  .btn-example:hover { background: rgba(52,152,219,0.3); }
  .btn-clear { background: rgba(231,76,60,0.2); color: #e74c3c; border: 1px solid rgba(231,76,60,0.3); }
  .btn-clear:hover { background: rgba(231,76,60,0.3); }
  #result { display: none; }
  .result-header { display: flex; align-items: center; gap: 16px; margin-bottom: 20px; }
  .tier-badge {
    font-size: 2.2rem; width: 114px; height: 64px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 16px; font-weight: 700;
  }
  .tier-badge.Premium { background: rgba(46,204,113,0.15); color: #2ecc71; }
  .tier-badge.Standard { background: rgba(52,152,219,0.15); color: #3498db; }
  .tier-badge.Basic { background: rgba(231,76,60,0.15); color: #e74c3c; }
  .tier-name { font-size: 1.6rem; font-weight: 700; }
  .tier-note { color: #999; font-size: 0.85rem; margin-top: 2px; }
  .comparison-table { width: 100%; border-collapse: collapse; margin: 16px 0; }
  .comparison-table th {
    text-align: left; padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,0.1);
    color: #999; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .comparison-table td { padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.9rem; }
  .status-above { color: #2ecc71; }
  .status-below { color: #e74c3c; }
  .status-at { color: #f39c12; }
  .tip-box {
    background: rgba(241,196,15,0.1); border: 1px solid rgba(241,196,15,0.2);
    border-radius: 10px; padding: 14px 18px; margin-top: 12px;
    font-size: 0.9rem; line-height: 1.5;
  }
  .tip-box strong { color: #f1c40f; }
  .examples-row { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
  .examples-row button { flex: none; padding: 8px 16px; font-size: 0.82rem; border-radius: 8px; }
  .spinner { display: none; text-align: center; padding: 20px; color: #3498db; }
  .footer { text-align: center; color: #555; font-size: 0.78rem; margin-top: 12px; }
  .footer a { color: #3498db; text-decoration: none; }
  @media (max-width: 700px) {
    .tooltip-text { left: 0; top: 100%; width: 100%; }
    .btn-row { flex-direction: column; }
  }
</style>
</head>
<body>
<div class="container">
  <h1>Research Center Quality Classifier</h1>
  <p class="subtitle">Classify research centers into Premium, Standard, or Basic tiers using ML</p>

  <div class="card">
    <div class="examples-row">
      <span style="color:#999;font-size:0.85rem;padding:8px 0;">Try an example:</span>
      <button class="btn-example" onclick="fillExample('premium')">Premium</button>
      <button class="btn-example" onclick="fillExample('standard')">Standard</button>
      <button class="btn-example" onclick="fillExample('basic')">Basic</button>
    </div>

    <form id="predictForm" onsubmit="handleSubmit(event)">
      <div class="field-group">
        <label>
          Internal Facilities Count
          <span class="tooltip-icon">?</span>
          <span class="tooltip-text">
            Number of internal facilities such as labs, testing units, and workstations.
            Higher values indicate a better-equipped center.
            <span class="range">Typical range: 1 – 11</span>
          </span>
        </label>
        <input type="number" id="internalFacilitiesCount" name="internalFacilitiesCount"
               placeholder="e.g. 9" step="1" min="0" required>
      </div>

      <div class="field-group">
        <label>
          Hospitals within 10km
          <span class="tooltip-icon">?</span>
          <span class="tooltip-text">
            Number of hospitals within a 10km radius.
            More nearby hospitals means better emergency and specialist access for trial participants.
            <span class="range">Typical range: 0 – 4</span>
          </span>
        </label>
        <input type="number" id="hospitals_10km" name="hospitals_10km"
               placeholder="e.g. 3" step="1" min="0" required>
      </div>

      <div class="field-group">
        <label>
          Pharmacies within 10km
          <span class="tooltip-icon">?</span>
          <span class="tooltip-text">
            Number of pharmacies within a 10km radius.
            Important for medication availability and dispensing during clinical trials.
            <span class="range">Typical range: 0 – 5</span>
          </span>
        </label>
        <input type="number" id="pharmacies_10km" name="pharmacies_10km"
               placeholder="e.g. 4" step="1" min="0" required>
      </div>

      <div class="field-group">
        <label>
          Facility Diversity Index
          <span class="tooltip-icon">?</span>
          <span class="tooltip-text">
            A score from 0 to 1 measuring how varied the types of nearby healthcare facilities are.
            1.0 = maximum diversity (many different types). 0.0 = no diversity.
            <span class="range">Range: 0.0 – 1.0</span>
          </span>
        </label>
        <input type="number" id="facilityDiversity_10km" name="facilityDiversity_10km"
               placeholder="e.g. 0.82" step="0.01" min="0" max="1" required>
      </div>

      <div class="field-group">
        <label>
          Facility Density Score
          <span class="tooltip-icon">?</span>
          <span class="tooltip-text">
            Approximate density of nearby healthcare facilities per area within 10km.
            Higher values mean more concentrated services around the center.
            <span class="range">Typical range: 0.05 – 0.70</span>
          </span>
        </label>
        <input type="number" id="facilityDensity_10km" name="facilityDensity_10km"
               placeholder="e.g. 0.45" step="0.01" min="0" required>
      </div>

      <div class="btn-row">
        <button type="submit" class="btn-predict" id="submitBtn">Classify Center</button>
        <button type="button" class="btn-clear" onclick="clearForm()">Clear</button>
      </div>
    </form>
  </div>

  <div class="spinner" id="spinner">Classifying...</div>

  <div class="card" id="result">
    <div class="result-header">
      <div class="tier-badge" id="tierBadge"></div>
      <div>
        <div class="tier-name" id="tierName"></div>
        <div class="tier-note" id="tierNote"></div>
      </div>
    </div>

    <table class="comparison-table">
      <thead>
        <tr><th>Feature</th><th>Your Value</th><th>Tier Average</th><th>Status</th></tr>
      </thead>
      <tbody id="comparisonBody"></tbody>
    </table>

    <div class="tip-box" id="tipBox"></div>
  </div>

  <div class="footer">
    Powered by K-Means Clustering &middot; <a href="/docs">API Docs</a> &middot; <a href="/redoc">ReDoc</a>
  </div>
</div>

<script>
const EXAMPLES = {
  premium:  { internalFacilitiesCount: 10, hospitals_10km: 4, pharmacies_10km: 5, facilityDiversity_10km: 0.88, facilityDensity_10km: 0.55 },
  standard: { internalFacilitiesCount: 5,  hospitals_10km: 2, pharmacies_10km: 2, facilityDiversity_10km: 0.55, facilityDensity_10km: 0.30 },
  basic:    { internalFacilitiesCount: 2,  hospitals_10km: 0, pharmacies_10km: 0, facilityDiversity_10km: 0.20, facilityDensity_10km: 0.10 },
};

function fillExample(tier) {
  const ex = EXAMPLES[tier];
  for (const [key, val] of Object.entries(ex)) {
    document.getElementById(key).value = val;
  }
}

function clearForm() {
  document.getElementById('predictForm').reset();
  document.getElementById('result').style.display = 'none';
}

async function handleSubmit(e) {
  e.preventDefault();
  const btn = document.getElementById('submitBtn');
  const spinner = document.getElementById('spinner');
  btn.disabled = true;
  spinner.style.display = 'block';
  document.getElementById('result').style.display = 'none';

  const body = {
    internalFacilitiesCount: parseFloat(document.getElementById('internalFacilitiesCount').value),
    hospitals_10km: parseFloat(document.getElementById('hospitals_10km').value),
    pharmacies_10km: parseFloat(document.getElementById('pharmacies_10km').value),
    facilityDiversity_10km: parseFloat(document.getElementById('facilityDiversity_10km').value),
    facilityDensity_10km: parseFloat(document.getElementById('facilityDensity_10km').value),
  };

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json();
      alert('Error: ' + (err.detail || JSON.stringify(err)));
      return;
    }
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    alert('Request failed: ' + err.message);
  } finally {
    btn.disabled = false;
    spinner.style.display = 'none';
  }
}

function renderResult(data) {
  const badge = document.getElementById('tierBadge');
  badge.textContent = data.tierIcon;
  badge.className = 'tier-badge ' + data.predictedCategory;

  document.getElementById('tierName').textContent = data.predictedCategory + ' Tier';
  document.getElementById('tierNote').textContent = data.confidence_note;

  const tbody = document.getElementById('comparisonBody');
  tbody.innerHTML = '';
  for (const c of data.comparison) {
    const statusClass = 'status-' + c.status;
    const arrow = c.status === 'above' ? '▲' : c.status === 'below' ? '▼' : '●';
    tbody.innerHTML += '<tr>'
      + '<td>' + c.feature + '</td>'
      + '<td>' + c.your_value + '</td>'
      + '<td>' + c.tier_average + '</td>'
      + '<td class="' + statusClass + '">' + arrow + ' ' + c.status + '</td>'
      + '</tr>';
  }

  document.getElementById('tipBox').innerHTML = '<strong>Tip:</strong> ' + data.tip;
  document.getElementById('result').style.display = 'block';
  document.getElementById('result').scrollIntoView({ behavior: 'smooth' });
}
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return {
        "status": "healthy",
        "model": "K-Means Research Center Classifier",
        "version": "1.0.0",
        "tiers_available": list(cluster_to_tier.values()),
    }


@app.get("/model-info", tags=["Info"])
def model_info():
    """
    Returns model metadata, feature descriptions with tooltips,
    tier benchmarks, and example inputs.
    """
    return {
        "model_type": "K-Means Clustering",
        "n_clusters": 3,
        "features": {
            name: {
                "label": info["label"],
                "tooltip": info["tooltip"],
                "range": info["range"],
                "unit": info["unit"],
            }
            for name, info in FEATURE_TOOLTIPS.items()
        },
        "tier_benchmarks": TIER_BENCHMARKS,
        "tier_mapping": {str(k): v for k, v in cluster_to_tier.items()},
        "tips": {
            "Premium": "This center is well-suited for complex, multi-phase clinical trials.",
            "Standard": "Suitable for routine trials. Consider investing in facilities to reach Premium.",
            "Basic": "Best for simple studies. Significant upgrades needed for complex trials.",
        },
    }


@app.get("/feature-help", tags=["Info"])
def feature_help():
    """
    Returns detailed help for every input feature — what it means,
    why it matters, and what values to expect.
    """
    return {
        name: {
            "label": info["label"],
            "description": info["tooltip"],
            "typical_range": info["range"],
            "unit": info["unit"],
            "why_it_matters": {
                "internalFacilitiesCount": "Centers with more internal facilities can run more concurrent studies and handle complex protocols.",
                "hospitals_10km": "Nearby hospitals are critical for managing serious adverse events during trials.",
                "pharmacies_10km": "Pharmacy access ensures reliable medication dispensing to trial participants.",
                "facilityDiversity_10km": "Diverse nearby facilities indicate a well-developed healthcare ecosystem.",
                "facilityDensity_10km": "Higher density means more healthcare options per area, improving participant convenience.",
            }.get(name, ""),
        }
        for name, info in FEATURE_TOOLTIPS.items()
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_quality(data: ResearchCenterInput):
    """
    Classify a research center into a quality tier.

    **How it works:**
    1. Your 5 input features are scaled using the same StandardScaler from training
    2. The K-Means model predicts which cluster the center belongs to
    3. The cluster is mapped to a human-readable tier (Premium / Standard / Basic)
    4. You get a comparison showing how your values stack up against the tier average

    **Tip:** Use the **GET /model-info** endpoint to see feature descriptions and tier benchmarks.
    """
    try:
        input_df = pd.DataFrame([data.model_dump()])
        input_df = input_df[selected_features]

        X_scaled = scaler.transform(input_df)
        cluster_label = int(kmeans.predict(X_scaled)[0])
        tier = cluster_to_tier.get(cluster_label, "Unknown")

        # Compute distance to assigned centroid for confidence context
        centroid = kmeans.cluster_centers_[cluster_label]
        dist = float(np.linalg.norm(X_scaled[0] - centroid))
        if dist < 1.0:
            confidence_note = "Strong fit — this center is close to the cluster center."
        elif dist < 2.0:
            confidence_note = "Good fit — clearly within this tier."
        else:
            confidence_note = "Borderline — this center is near the edge of this tier."

        # Compare input values to tier benchmarks
        benchmarks = TIER_BENCHMARKS.get(tier, {})
        comparison = []
        for feat in selected_features:
            your_val = round(getattr(data, feat), 2)
            avg_val = round(benchmarks.get(feat, 0), 2)
            if your_val > avg_val * 1.05:
                status = "above"
            elif your_val < avg_val * 0.95:
                status = "below"
            else:
                status = "at"
            comparison.append(FeatureComparison(
                feature=FEATURE_TOOLTIPS[feat]["label"],
                your_value=your_val,
                tier_average=avg_val,
                status=status,
            ))

        # Generate actionable tip
        below_features = [c.feature for c in comparison if c.status == "below"]
        if tier == "Premium" and not below_features:
            tip = "Excellent — this center exceeds or meets Premium benchmarks across the board. Ideal for complex clinical trials."
        elif tier == "Premium":
            tip = f"Classified as Premium, but {', '.join(below_features)} is below average for this tier. Consider improvements in these areas."
        elif tier == "Standard":
            tip = f"To reach Premium, focus on improving: {', '.join(below_features) if below_features else 'all metrics proportionally'}. The biggest gaps are usually in internal facilities and hospital access."
        else:
            tip = "This center is classified as Basic. To improve, prioritise adding internal facilities and choosing locations with better healthcare access."

        logger.info(f"Prediction: {data.model_dump()} -> Cluster {cluster_label} -> {tier} (dist={dist:.2f})")

        cfg = TIER_CONFIG.get(tier, TIER_CONFIG["Unknown"])
        return PredictionResponse(
            predictedCluster=cluster_label,
            predictedCategory=tier,
            tierIcon=cfg["icon"],
            confidence_note=confidence_note,
            comparison=comparison,
            tip=tip,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Batch prediction — classify multiple centers at once

class BatchInput(BaseModel):
    """List of research centers for batch classification."""
    centers: list[ResearchCenterInput] = Field(
        description="List of research center inputs to classify"
    )


class BatchResultItem(BaseModel):
    index: int
    predictedCategory: str
    predictedCluster: int


class BatchResponse(BaseModel):
    total: int
    results: list[BatchResultItem]
    summary: dict[str, int]


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(data: BatchInput):
    """
    Classify multiple research centers in a single request.
    Useful for evaluating a portfolio of candidate sites at once.

    Returns individual predictions plus a summary count per tier.
    """
    try:
        rows = [center.model_dump() for center in data.centers]
        input_df = pd.DataFrame(rows)[selected_features]
        X_scaled = scaler.transform(input_df)
        labels = kmeans.predict(X_scaled)

        results = []
        tier_counts = {"Premium": 0, "Standard": 0, "Basic": 0}
        for i, label in enumerate(labels):
            tier = cluster_to_tier.get(int(label), "Unknown")
            results.append(BatchResultItem(
                index=i,
                predictedCategory=tier,
                predictedCluster=int(label),
            ))
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        logger.info(f"Batch prediction: {len(data.centers)} centers -> {tier_counts}")

        return BatchResponse(
            total=len(results),
            results=results,
            summary=tier_counts,
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
