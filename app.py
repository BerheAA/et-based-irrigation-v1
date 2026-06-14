import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import altair as alt
from datetime import date, datetime, timedelta

# -------------------------
# CONFIG
# -------------------------

st.set_page_config(
    page_title="ET based Irrigation Scheduling",
    layout="wide"
)

# Soft background theme similar to Nevada / New Mexico dashboard
st.markdown(
    """
    <style>
    html, body, [data-testid="stApp"] {
        background-color: #f5f5f5;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
    }
    .metric-card {
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        background-color: #ffffff;
        box-shadow: 0 0 8px rgba(0,0,0,0.05);
        margin-bottom: 0.5rem;
    }
    .section-header {
        padding: 0.4rem 0.8rem;
        border-radius: 0.5rem;
        display: inline-block;
        font-weight: 600;
        background-color: #e3f2fd;
        margin-bottom: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENET_POINT_URL = "https://openet-api.org/raster/timeseries/point"
SSURGO_SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"

# -------------------------
# REFERENCE DATA
# -------------------------

KS_EXAMPLE_LOCATIONS = {
    "Colby (NW Kansas irrigated corn/wheat)": (39.3953, -101.0524),
    "Garden City (SW Kansas High Plains)": (37.9717, -100.8727),
    "Dodge City": (37.7528, -100.0171),
    "Scott City": (38.4825, -100.9071),
    "Hays (Central KS)": (38.8792, -99.3268),
    "Great Bend": (38.3645, -98.7640),
    "Salina": (38.8403, -97.6114),
    "Manhattan (NE/Central research station)": (39.1836, -96.5717),
    "McPherson": (38.3708, -97.6648),
    "Wichita (South Central)": (37.6872, -97.3301),
    "Parsons (SE Kansas)": (37.3409, -95.2591),
    "User-defined": (39.1836, -96.5717),
}

CROP_PARAMS = {
    "Corn (grain)": {
        "kc": 1.15,
        "root_depth_m": 1.2,
        "yield_potential_mmy": 15000,
        "ky": 1.25,
        "season_length_days": 150,
    },
    "Corn (silage)": {
        "kc": 1.15,
        "root_depth_m": 1.2,
        "yield_potential_mmy": 25000,
        "ky": 1.1,
        "season_length_days": 145,
    },
    "Grain sorghum": {
        "kc": 1.0,
        "root_depth_m": 1.3,
        "yield_potential_mmy": 9000,
        "ky": 1.0,
        "season_length_days": 130,
    },
    "Soybean": {
        "kc": 1.05,
        "root_depth_m": 1.1,
        "yield_potential_mmy": 4500,
        "ky": 1.1,
        "season_length_days": 145,
    },
    "Winter wheat": {
        "kc": 1.05,
        "root_depth_m": 1.0,
        "yield_potential_mmy": 7000,
        "ky": 1.0,
        "season_length_days": 220,
    },
    "Alfalfa": {
        "kc": 1.1,
        "root_depth_m": 1.3,
        "yield_potential_mmy": 14000,
        "ky": 1.1,
        "season_length_days": 210,
    },
    "Pasture / Grass": {
        "kc": 0.95,
        "root_depth_m": 0.8,
        "yield_potential_mmy": 9000,
        "ky": 1.0,
        "season_length_days": 180,
    },
}

SOIL_TYPES = {
    "Sandy loam": {
        "description": "Lower water holding, common in parts of western KS",
        "TAW_mm_per_m": 110,
    },
    "Loam": {
        "description": "Balanced soil with moderate water storage",
        "TAW_mm_per_m": 140,
    },
    "Silt loam": {
        "description": "Typical of central and eastern KS, good water holding",
        "TAW_mm_per_m": 160,
    },
    "Clay loam": {
        "description": "Heavier soils with high water holding",
        "TAW_mm_per_m": 170,
    },
}

STRATEGIES = {
    "Full irrigation (no intentional stress)": 0.45,
    "Moderate deficit (light stress allowed)": 0.6,
    "Severe deficit (only critical irrigations)": 0.8,
}

IRRIGATION_SYSTEMS = {
    "Center pivot": 0.85,
    "Sprinkler (solid set/line)": 0.80,
    "Surface / Furrow": 0.65,
    "Drip": 0.90,
}

# Kansas / DSSAT-style automatic irrigation trigger used here for center
# pivot/sprinkler scheduling in major cereal and legume crops:
# apply a fixed event depth (commonly 25 mm) when the soil-water deficit
# reaches 50% of the maximum available water in the management depth.
# The management depth used here is the top 30 cm, matching the user's DSSAT
# automatic-irrigation interpretation.
MANAGEMENT_DEPTH_M = 0.30
KANSAS_TRIGGER_FRACTION = 0.50

CEREAL_AND_LEGUME_CROPS = {
    "Corn (grain)",
    "Corn (silage)",
    "Grain sorghum",
    "Soybean",
    "Winter wheat",
}

SPRINKLER_PIVOT_SYSTEMS = {
    "Center pivot",
    "Sprinkler (solid set/line)",
}

# -------------------------
# WEATHER FUNCTIONS
# -------------------------


def safe_empty_weather():
    """Standard empty weather dataframe used when API/upload data are unavailable."""
    return pd.DataFrame(columns=["time", "precip_mm", "et0_mm", "tmax_c", "tmin_c"])


def clean_numeric_series(series, default=0.0):
    """Convert a series to numeric values and remove NaN/inf safely."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.fillna(default)


def show_small_table(df):
    """Render small tables as plain HTML to avoid fragile Streamlit dataframe JS chunks."""
    if df is None or df.empty:
        st.info("No table data available.")
        return

    html = df.to_html(index=False, border=0, classes="simple-table")
    st.markdown(
        """
        <style>
        .simple-table {
            border-collapse: collapse;
            width: 100%;
            background: white;
            font-size: 0.95rem;
        }
        .simple-table th, .simple-table td {
            border: 1px solid #dddddd;
            padding: 0.45rem 0.6rem;
            text-align: left;
        }
        .simple-table th {
            background-color: #e3f2fd;
            font-weight: 600;
        }
        </style>
        """
        + html,
        unsafe_allow_html=True,
    )


def request_json_with_retry(
    url,
    params=None,
    method="GET",
    data=None,
    json_payload=None,
    headers=None,
    timeout=20,
    max_retries=3,
):
    """
    Request JSON with retry/backoff.

    Supports both:
    - GET requests for Open-Meteo
    - POST JSON requests for OpenET

    This reduces dashboard failure when APIs temporarily return 429 or 5xx errors.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            if method.upper() == "POST":
                response = requests.post(
                    url,
                    data=data,
                    json=json_payload,
                    headers=headers,
                    timeout=timeout,
                )
            else:
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                )

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = Exception(
                    f"{response.status_code} API error: {response.text[:250]}"
                )
                time.sleep(2 * (attempt + 1))
                continue

            response.raise_for_status()
            return response.json(), None

        except Exception as e:
            last_error = e
            time.sleep(2 * (attempt + 1))

    return None, last_error

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_archive_weather(lat, lon, start_date, end_date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": [
            "precipitation_sum",
            "et0_fao_evapotranspiration",
            "temperature_2m_max",
            "temperature_2m_min",
        ],
        "timezone": "America/Chicago",
    }
    data, err = request_json_with_retry(OPEN_METEO_ARCHIVE_URL, params=params, timeout=20)
    if err is not None or not data or "daily" not in data:
        st.warning(f"Could not retrieve historical weather data: {err}")
        return safe_empty_weather()

    daily = data["daily"]
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df.rename(
        columns={
            "precipitation_sum": "precip_mm",
            "et0_fao_evapotranspiration": "et0_mm",
            "temperature_2m_max": "tmax_c",
            "temperature_2m_min": "tmin_c",
        },
        inplace=True,
    )
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_forecast_weather(lat, lon, start_date, end_date):
    from datetime import date as _date
    days_ahead = (end_date - _date.today()).days
    if days_ahead <= 0:
        return pd.DataFrame()

    days_ahead = min(days_ahead, 16)

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "precipitation_sum",
            "et0_fao_evapotranspiration",
            "temperature_2m_max",
            "temperature_2m_min",
        ],
        "timezone": "America/Chicago",
        "forecast_days": days_ahead,
        "past_days": 0,
    }

    data, err = request_json_with_retry(OPEN_METEO_FORECAST_URL, params=params, timeout=20)
    if err is not None or not data or "daily" not in data:
        st.warning(f"Could not retrieve forecast weather data: {err}")
        return safe_empty_weather()

    daily = data["daily"]
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df.rename(
        columns={
            "precipitation_sum": "precip_mm",
            "et0_fao_evapotranspiration": "et0_mm",
            "temperature_2m_max": "tmax_c",
            "temperature_2m_min": "tmin_c",
        },
        inplace=True,
    )
    df = df[df["time"].dt.date >= start_date]
    df = df[df["time"].dt.date <= end_date]
    return df


def get_season_weather(lat, lon, planting_date, season_length):
    start_date = planting_date
    end_date = planting_date + timedelta(days=season_length)

    today = date.today()
    hist_end = min(end_date, today - timedelta(days=1))
    hist_df = pd.DataFrame()
    if hist_end >= start_date:
        hist_df = fetch_archive_weather(lat, lon, start_date, hist_end)

    fc_df = pd.DataFrame()
    if end_date >= today:
        fc_df = fetch_forecast_weather(lat, lon, max(start_date, today), end_date)

    if not hist_df.empty and not fc_df.empty:
        df = pd.concat([hist_df, fc_df], ignore_index=True)
    elif not hist_df.empty:
        df = hist_df
    else:
        df = fc_df

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.sort_values("time")
    df = df[df["time"].dt.date >= start_date]
    df = df[df["time"].dt.date <= end_date]
    df.reset_index(drop=True, inplace=True)
    return df


def generate_simple_eto_weather(
    planting_date,
    season_length,
    base_eto_mm=5.0,
    seasonal_amp_mm=1.0,
    mean_precip_mm=2.0,
    rain_probability=0.30,
):
    """Generate a simple synthetic ET0 and rainfall pattern for demonstration."""
    dates = [planting_date + timedelta(days=i) for i in range(season_length)]
    doy = np.array([d.timetuple().tm_yday for d in dates])

    et0 = base_eto_mm + seasonal_amp_mm * np.sin(2 * np.pi * (doy - 190) / 365.0)
    et0 = np.maximum(et0, 0.0)

    rng = np.random.default_rng(42)
    rain_flag = rng.uniform(size=len(dates)) < rain_probability
    precip = np.where(rain_flag, rng.gamma(shape=1.5, scale=mean_precip_mm / 1.5), 0.0)

    tmean = 23 + 7 * np.sin(2 * np.pi * (doy - 200) / 365.0)
    tmax = tmean + 6
    tmin = tmean - 6

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(dates),
            "precip_mm": precip,
            "et0_mm": et0,
            "tmax_c": tmax,
            "tmin_c": tmin,
        }
    )
    return df


def load_climate_from_csv(uploaded_file, planting_date, season_length):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV file: {e}")
        return pd.DataFrame()

    required_cols = ["date", "precip_mm", "et0_mm", "tmax_c", "tmin_c"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        return pd.DataFrame()

    try:
        df["time"] = pd.to_datetime(df["date"])
    except Exception as e:
        st.error(f"Could not parse 'date' column: {e}")
        return pd.DataFrame()

    df = df.sort_values("time")
    start_date = planting_date
    end_date = planting_date + timedelta(days=season_length)
    df = df[(df["time"].dt.date >= start_date) & (df["time"].dt.date <= end_date)]

    if df.empty:
        st.error("No climate data in CSV covers the requested season.")
        return pd.DataFrame()

    return df[["time", "precip_mm", "et0_mm", "tmax_c", "tmin_c"]].copy()


# -------------------------
# OPENET FUNCTIONS
# -------------------------


def get_openet_api_key():
    """Read OpenET API key from Streamlit secrets or environment variable."""
    try:
        key = st.secrets.get("OPENET_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("OPENET_API_KEY", "")


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_openet_actual_et(lat, lon, start_date, end_date, model="Ensemble", interval="daily"):
    """
    Retrieve OpenET satellite-based actual evapotranspiration for a point.

    OpenET returns actual ET, not reference ET0. Therefore, this value should be
    used directly in the water balance and should NOT be multiplied by Kc again.

    Returns a dataframe with columns:
        time, openet_actual_et_mm
    """
    empty = pd.DataFrame(columns=["time", "openet_actual_et_mm"])

    api_key = get_openet_api_key()
    if not api_key:
        st.warning(
            "OpenET API key is missing. Add OPENET_API_KEY in Streamlit Cloud secrets "
            "or set it as an environment variable. The app will fall back to Open-Meteo."
        )
        return empty

    if end_date < start_date:
        return empty

    headers = {
        "Authorization": api_key,
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    payload = {
        "date_range": [
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        ],
        "interval": interval,
        "geometry": [float(lon), float(lat)],
        "model": model,
        "variable": "ET",
        "reference_et": "gridMET",
        "units": "mm",
        "file_format": "JSON",
    }

    data, err = request_json_with_retry(
        OPENET_POINT_URL,
        method="POST",
        json_payload=payload,
        headers=headers,
        timeout=60,
        max_retries=3,
    )

    if err is not None or data is None:
        # OpenET sometimes returns temporary 5xx server errors under load.
        # Do not show a scary raw API error or stop the dashboard.
        # Return an empty dataframe so apply_et_method() can use the safe Open-Meteo fallback.
        return empty

    parsed = parse_openet_timeseries(data, start_date, end_date)

    # If OpenET responded but the structure was not usable, still fail safely.
    if parsed is None or parsed.empty:
        return empty

    return parsed


def parse_openet_timeseries(data, start_date, end_date):
    """Parse likely OpenET JSON response structures into a clean dataframe."""
    empty = pd.DataFrame(columns=["time", "openet_actual_et_mm"])

    records = None
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key in ["data", "timeseries", "results", "features"]:
            if key in data:
                records = data[key]
                break
        if records is None:
            records = data
    else:
        return empty

    # GeoJSON-like response
    if isinstance(records, list) and records and isinstance(records[0], dict) and "properties" in records[0]:
        records = [r.get("properties", {}) for r in records]

    try:
        df = pd.DataFrame(records)
    except Exception:
        return empty

    if df.empty:
        return empty

    # Some API responses may use nested dictionaries. Flatten one level if needed.
    if any(isinstance(v, dict) for v in df.iloc[0].values):
        df = pd.json_normalize(records)

    date_candidates = [
        "time", "date", "dt", "start_date", "end_date", "start", "system:time_start"
    ]
    et_candidates = [
        "ET", "et", "value", "mean", "et_mm", "openet_et", "properties.ET", "properties.et"
    ]

    date_col = next((c for c in date_candidates if c in df.columns), None)
    et_col = next((c for c in et_candidates if c in df.columns), None)

    # If OpenET returns columns by model name, look for a numeric ET-like column.
    if et_col is None:
        numeric_cols = [c for c in df.columns if c != date_col]
        for c in numeric_cols:
            if str(c).lower() in ["ensemble", "ssebop", "eemetric", "geeSEBAL".lower(), "ptjpl", "sims", "disalexi"]:
                et_col = c
                break

    if date_col is None or et_col is None:
        # Response format was not usable. Return empty and let the dashboard fallback safely.
        return empty

    out = df[[date_col, et_col]].copy()
    out.columns = ["time", "openet_actual_et_mm"]
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out["openet_actual_et_mm"] = pd.to_numeric(out["openet_actual_et_mm"], errors="coerce")
    out = out.dropna(subset=["time", "openet_actual_et_mm"])

    if out.empty:
        return empty

    # If monthly values are returned, distribute monthly ET evenly to daily values
    # because the irrigation engine runs with a daily water balance.
    median_gap = out["time"].sort_values().diff().dt.days.median()
    if pd.notna(median_gap) and median_gap > 7:
        daily_rows = []
        for _, row in out.iterrows():
            month_start = row["time"].date().replace(day=1)
            next_month = (pd.Timestamp(month_start) + pd.offsets.MonthBegin(1)).date()
            month_end = min(next_month - timedelta(days=1), end_date)
            current_start = max(month_start, start_date)
            n_days = (month_end - current_start).days + 1
            if n_days <= 0:
                continue
            daily_value = row["openet_actual_et_mm"] / n_days
            for i in range(n_days):
                daily_rows.append(
                    {
                        "time": pd.Timestamp(current_start + timedelta(days=i)),
                        "openet_actual_et_mm": daily_value,
                    }
                )
        out = pd.DataFrame(daily_rows)

    out = out[(out["time"].dt.date >= start_date) & (out["time"].dt.date <= end_date)]
    out = out.sort_values("time").drop_duplicates(subset=["time"])
    out.reset_index(drop=True, inplace=True)
    return out


def apply_et_method(df_weather, lat, lon, planting_date, season_length, crop_name, et_method, openet_model, openet_interval):
    """
    Add et_for_irrigation_mm and et_source columns based on selected ET method.

    Methods:
    - Open-Meteo ET0 x Kc: weather-based crop ET estimate
    - OpenET actual ET only: satellite actual ET for historical dates only
    - Hybrid: OpenET historical actual ET + Open-Meteo forecast ET0 x Kc
    """
    if df_weather is None or df_weather.empty:
        return df_weather

    df = df_weather.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["time"]).copy()

    crop_kc = CROP_PARAMS[crop_name]["kc"]
    df["et0_mm"] = clean_numeric_series(df["et0_mm"], default=0.0)
    df["precip_mm"] = clean_numeric_series(df["precip_mm"], default=0.0)
    df["openmeteo_etc_mm"] = (df["et0_mm"] * crop_kc).clip(lower=0)
    df["et_for_irrigation_mm"] = df["openmeteo_etc_mm"]
    df["et_source"] = "Open-Meteo ET0 x Kc"

    if et_method == "Open-Meteo ET0 x Kc":
        return df

    today = date.today()
    season_start = planting_date
    season_end = planting_date + timedelta(days=season_length)
    openet_end = min(season_end, today - timedelta(days=1))

    if openet_end < season_start:
        st.info(
            "OpenET actual ET is available for past/recent operational dates only. "
            "Because this selected season is entirely in the future, the app is using Open-Meteo ET0 × Kc forecast."
        )
        return df

    if not get_openet_api_key():
        st.info(
            "OpenET is optional and is not enabled because no OPENET_API_KEY is configured. "
            "The dashboard is running normally with Open-Meteo ET0 x Kc. "
            "To enable OpenET, add OPENET_API_KEY in Streamlit Cloud secrets or in local .streamlit/secrets.toml."
        )
        df["et_source"] = "Open-Meteo fallback: OpenET API key missing"
        return df

    df_openet = fetch_openet_actual_et(
        lat=lat,
        lon=lon,
        start_date=season_start,
        end_date=openet_end,
        model=openet_model,
        interval=openet_interval,
    )

    if df_openet.empty:
        st.info(
            "OpenET actual ET is temporarily unavailable or returned no usable values. "
            "The dashboard is continuing normally with Open-Meteo ET0 x Kc fallback."
        )
        df["et_source"] = "Open-Meteo fallback: OpenET unavailable"
        return df

    df_openet = df_openet.copy()
    df_openet["time"] = pd.to_datetime(df_openet["time"], errors="coerce").dt.normalize()
    df_openet["openet_actual_et_mm"] = clean_numeric_series(df_openet["openet_actual_et_mm"], default=np.nan)
    df_openet = df_openet.dropna(subset=["time", "openet_actual_et_mm"])
    df = df.merge(df_openet, on="time", how="left")

    historical_mask = df["time"].dt.date <= openet_end
    has_openet = historical_mask & df["openet_actual_et_mm"].notna()

    if et_method == "OpenET actual ET only":
        df.loc[historical_mask, "et_for_irrigation_mm"] = df.loc[historical_mask, "openet_actual_et_mm"]
        df.loc[has_openet, "et_source"] = "OpenET actual ET"
        missing_openet = historical_mask & df["openet_actual_et_mm"].isna()
        df.loc[missing_openet, "et_source"] = "Open-Meteo fallback: OpenET missing"
        if season_end >= today:
            st.info(
                "OpenET actual ET is used for all available past/recent operational dates. "
                "Future dates use Open-Meteo ET0 × Kc forecast because OpenET does not provide future ET forecasts."
            )

    elif et_method == "Hybrid: OpenET historical + Open-Meteo forecast":
        df.loc[has_openet, "et_for_irrigation_mm"] = df.loc[has_openet, "openet_actual_et_mm"]
        df.loc[has_openet, "et_source"] = "OpenET historical actual ET"
        # Future dates and missing OpenET dates keep the Open-Meteo ET0 x Kc estimate.

    return df

# -------------------------
# SSURGO SOIL LOOKUP (BETA)
# -------------------------

def lookup_ssurgo_soil(lat, lon):
    """
    Very simple SSURGO lookup using NRCS SDA Tabular service (beta).
    Returns dominant component name and taxonomic order, if available.
    """
    sql = f"""
        SELECT TOP 1 c.compname, c.taxorder
        FROM mapunit AS mu
        INNER JOIN component AS c ON c.mukey = mu.mukey
        WHERE mu.mukey IN (
            SELECT TOP 1 mukey
            FROM SDA_Get_Mukey_from_intersection_with_WktWgs84(
                'POINT({lon} {lat})'
            )
        )
        ORDER BY c.comppct_r DESC
    """
    payload = {"format": "JSON+COLUMNS", "query": sql}
    try:
        r = requests.post(SSURGO_SDA_URL, data=payload, timeout=25)
        r.raise_for_status()
        data = r.json()
        if "Table" in data and len(data["Table"]) > 0:
            row = data["Table"][0]
            return {
                "compname": row.get("compname", None),
                "taxorder": row.get("taxorder", None),
            }
        else:
            return None
    except Exception as e:
        st.warning(f"SSURGO lookup failed: {e}")
        return None


# -------------------------
# IRRIGATION ENGINE
# -------------------------


def simulate_irrigation(
    df_weather,
    crop_name,
    soil_name,
    strategy_label,
    irrigation_system=None,
    irrigation_efficiency=0.85,
    rainfall_efficiency=0.8,
    irrigation_application_mm=25.0,
):
    """
    Daily water-balance irrigation scheduling.

    Main operational correction:
    - Center-pivot/sprinkler irrigation for cereal and legume crops is triggered
      using the top 30 cm management depth, not the full crop root zone.
    - The trigger is 50% depletion of the available water in that management depth.
    - Each event applies the selected gross event depth, typically 25 mm.
    """

    if df_weather is None or df_weather.empty:
        return None, None, safe_empty_weather()

    crop = CROP_PARAMS[crop_name]
    soil = SOIL_TYPES[soil_name]

    kc = crop["kc"]
    root_depth = crop["root_depth_m"]
    taw_per_m = soil["TAW_mm_per_m"]

    # Full crop-root-zone TAW is retained for seasonal water-balance summaries.
    taw = taw_per_m * root_depth

    # Management-depth TAW controls automatic irrigation scheduling.
    management_depth_m = MANAGEMENT_DEPTH_M
    management_taw_mm = taw_per_m * management_depth_m

    strategy_depletion_fraction = STRATEGIES[strategy_label]
    use_kansas_50pct_trigger = (
        irrigation_system in SPRINKLER_PIVOT_SYSTEMS
        and crop_name in CEREAL_AND_LEGUME_CROPS
    )

    if use_kansas_50pct_trigger:
        allowable_depletion_fraction = KANSAS_TRIGGER_FRACTION
        trigger_rule = (
            "Kansas/DSSAT-style 50% trigger: irrigate when deficit reaches "
            "50% of available water in the top 30 cm management depth."
        )
    else:
        allowable_depletion_fraction = strategy_depletion_fraction
        trigger_rule = (
            "Strategy-based trigger from selected irrigation strategy, calculated "
            "over the top 30 cm management depth for scheduling."
        )

    trigger_depletion_mgmt_mm = management_taw_mm * allowable_depletion_fraction
    trigger_storage_mgmt_mm = management_taw_mm - trigger_depletion_mgmt_mm
    trigger_depletion_pct = allowable_depletion_fraction * 100.0
    trigger_storage_pct = 100.0 - trigger_depletion_pct

    df = df_weather.copy()
    df["date"] = df["time"].dt.date

    # Use selected ET if present; otherwise calculate ETc from ET0 x Kc.
    if "et_for_irrigation_mm" in df.columns:
        df["etc_mm"] = pd.to_numeric(df["et_for_irrigation_mm"], errors="coerce")
    else:
        df["etc_mm"] = pd.to_numeric(df["et0_mm"], errors="coerce") * kc

    df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce").fillna(0.0)
    df["etc_mm"] = pd.to_numeric(df["etc_mm"], errors="coerce").fillna(0.0)
    df["eff_precip_mm"] = df["precip_mm"] * rainfall_efficiency

    gross_event_depth = max(float(irrigation_application_mm), 0.0)

    # Root-zone storage supports seasonal ET deficit summaries.
    root_storage = taw

    # Management-layer storage controls irrigation trigger and chart display.
    management_storage = management_taw_mm

    irrigations = []
    root_storage_list = []
    root_deficit_list = []
    management_storage_list = []
    management_deficit_list = []
    management_storage_pct_list = []
    management_deficit_pct_list = []
    actual_et_list = []
    et_deficit_list = []
    irrigation_applied_list = []
    effective_irrigation_applied_list = []

    etc_cum = 0.0
    actual_et_cum = 0.0
    et_deficit_cum = 0.0

    for _, row in df.iterrows():
        etc = max(float(row["etc_mm"]), 0.0)
        p_eff = max(float(row["eff_precip_mm"]), 0.0)

        # Rainfall enters both the full root zone and top 30 cm management layer.
        root_storage = min(taw, root_storage + p_eff)
        management_storage = min(management_taw_mm, management_storage + p_eff)

        # ET demand depletes both storages. The root-zone storage is used for
        # seasonal ET-demand-met summaries; the management storage is used for
        # automatic irrigation triggering and percent display.
        actual_et = min(root_storage, etc)
        root_storage = max(0.0, root_storage - actual_et)
        management_storage = max(0.0, management_storage - etc)

        et_deficit = max(0.0, etc - actual_et)

        management_deficit_before_irrigation = management_taw_mm - management_storage
        root_deficit_before_irrigation = taw - root_storage
        irrigation_mm = 0.0
        irrigation_effective_mm = 0.0

        # Trigger irrigation only after the management-depth deficit reaches
        # the selected threshold. For Kansas pivot/sprinkler cereal/legume
        # cases, this is exactly 50% of available water in the top 30 cm.
        if management_deficit_before_irrigation >= trigger_depletion_mgmt_mm and gross_event_depth > 0:
            irrigation_mm = gross_event_depth
            irrigation_effective_mm = irrigation_mm * irrigation_efficiency

            root_storage = min(taw, root_storage + irrigation_effective_mm)
            management_storage = min(management_taw_mm, management_storage + irrigation_effective_mm)

            irrigations.append(
                {
                    "date": row["date"],
                    "irrigation_mm": irrigation_mm,
                    "effective_irrigation_mm": irrigation_effective_mm,
                    "management_depth_m": management_depth_m,
                    "deficit_before_pct": 100.0 * management_deficit_before_irrigation / management_taw_mm if management_taw_mm > 0 else 0.0,
                    "deficit_after_pct": 100.0 * (management_taw_mm - management_storage) / management_taw_mm if management_taw_mm > 0 else 0.0,
                    "storage_after_pct": 100.0 * management_storage / management_taw_mm if management_taw_mm > 0 else 0.0,
                }
            )

        root_storage_list.append(root_storage)
        root_deficit_list.append(taw - root_storage)
        management_storage_list.append(management_storage)
        management_deficit_list.append(management_taw_mm - management_storage)
        management_storage_pct_list.append(
            100.0 * management_storage / management_taw_mm if management_taw_mm > 0 else 0.0
        )
        management_deficit_pct_list.append(
            100.0 * (management_taw_mm - management_storage) / management_taw_mm if management_taw_mm > 0 else 0.0
        )
        irrigation_applied_list.append(irrigation_mm)
        effective_irrigation_applied_list.append(irrigation_effective_mm)
        actual_et_list.append(actual_et)
        et_deficit_list.append(et_deficit)

        etc_cum += etc
        actual_et_cum += actual_et
        et_deficit_cum += et_deficit

    df["irrigation_applied_mm"] = irrigation_applied_list
    df["effective_irrigation_applied_mm"] = effective_irrigation_applied_list

    # Root-zone values are retained for continuity and seasonal summaries.
    df["soil_storage_mm"] = root_storage_list
    df["deficit_mm"] = root_deficit_list

    # Management-depth values drive the chart and automatic irrigation trigger.
    df["management_storage_mm"] = management_storage_list
    df["management_deficit_mm"] = management_deficit_list
    df["management_storage_pct"] = management_storage_pct_list
    df["management_deficit_pct"] = management_deficit_pct_list

    df["actual_et_mm"] = actual_et_list
    df["et_deficit_mm"] = et_deficit_list

    irrigation_df = pd.DataFrame(irrigations)

    total_irrigation_mm = (
        0.0 if irrigation_df is None or irrigation_df.empty
        else float(irrigation_df["irrigation_mm"].sum())
    )

    n_irrigations = (
        0 if irrigation_df is None or irrigation_df.empty
        else len(irrigation_df)
    )

    avg_irrigation_event_mm = (
        0.0 if irrigation_df is None or irrigation_df.empty
        else float(irrigation_df["irrigation_mm"].mean())
    )

    et_demand_met_pct = 100.0 * actual_et_cum / etc_cum if etc_cum > 0 else 100.0
    et_demand_met_pct = float(np.clip(et_demand_met_pct, 0.0, 100.0))

    summary = {
        "taw_mm": taw,
        "management_depth_m": management_depth_m,
        "management_taw_mm": management_taw_mm,
        "trigger_depletion_mm": trigger_depletion_mgmt_mm,
        "trigger_storage_mm": trigger_storage_mgmt_mm,
        "trigger_depletion_fraction": allowable_depletion_fraction,
        "trigger_depletion_pct": trigger_depletion_pct,
        "trigger_storage_pct": trigger_storage_pct,
        "trigger_rule": trigger_rule,
        "irrigation_application_mm": gross_event_depth,
        "total_etc_mm": etc_cum,
        "actual_et_mm": actual_et_cum,
        "total_deficit_mm": et_deficit_cum,
        "et_demand_met_pct": et_demand_met_pct,
        "n_irrigations": n_irrigations,
        "total_irrigation_mm": total_irrigation_mm,
        "avg_irrigation_event_mm": avg_irrigation_event_mm,
    }

    return irrigation_df, summary, df


# -------------------------
# MAIN APP LAYOUT
# -------------------------

st.title("ET based Irrigation Scheduling")
st.caption(
    "Prototype decision support tool for ET-based irrigation scheduling and water planning "
    "across Kansas (for demonstration and Extension use)."
)

st.markdown("---")

# SIDEBAR: LOCATION, CROP, CLIMATE, IRRIGATION
with st.sidebar:
    st.header("1. Location & Crop Setup")

    loc_name = st.selectbox(
        "Select location",
        options=list(KS_EXAMPLE_LOCATIONS.keys()),
        index=1,
        help="Choose a representative location or 'User-defined' and adjust coordinates.",
    )

    base_lat, base_lon = KS_EXAMPLE_LOCATIONS[loc_name]
    lat = st.number_input("Latitude (°N)", value=float(base_lat), format="%.4f")
    lon = st.number_input("Longitude (°E)", value=float(base_lon), format="%.4f")

    today = date.today()
    default_planting = date(today.year, 4, 15)
    planting_date = st.date_input(
        "Planting / emergence date",
        value=default_planting,
        help="Approximate start of main irrigation season.",
    )

    crop_name = st.selectbox("Crop", options=list(CROP_PARAMS.keys()), index=0)

    st.header("2. Soil & SSURGO (beta)")
    soil_name = st.selectbox(
        "Dominant soil type (manual)",
        options=list(SOIL_TYPES.keys()),
        index=1,
        help="Generic texture class used to estimate soil water holding capacity.",
    )

    use_ssurgo = st.checkbox(
        "Try SSURGO soil lookup from GPS (beta)",
        value=False,
        help="Queries NRCS Soil Data Access by coordinates and reports dominant soil component.",
    )
    if use_ssurgo:
        if st.button("Lookup SSURGO soil at this point"):
            ssurgo_info = lookup_ssurgo_soil(lat, lon)
            if ssurgo_info:
                st.success(
                    f"SSURGO dominant component: {ssurgo_info.get('compname', 'N/A')} "
                    f"(Taxorder: {ssurgo_info.get('taxorder', 'N/A')})"
                )
            else:
                st.warning("No SSURGO component found for this point (or request failed).")

    st.header("3. Climate Data / ETo options")

    climate_source = st.radio(
        "Climate / ET₀ source",
        options=[
            "Open-Meteo (automatic)",
            "Simple ET₀ pattern (demo)",
            "Upload daily climate CSV",
        ],
        index=0,
        help="Choose how to provide daily ET₀ and rainfall to the irrigation engine.",
    )

    et_method = "Open-Meteo ET0 x Kc"
    openet_model = "Ensemble"
    openet_interval = "monthly"

    if climate_source == "Open-Meteo (automatic)":
        et_method = st.selectbox(
            "ET method used by irrigation engine",
            options=[
                "Open-Meteo ET0 x Kc",
                "Hybrid: OpenET historical + Open-Meteo forecast",
                "OpenET actual ET only",
            ],
            index=0,
            help=(
                "OpenET is satellite-based actual ET for historical dates. "
                "Open-Meteo ET0 x Kc is retained for forecast dates and as a fallback."
            ),
        )

        if et_method != "Open-Meteo ET0 x Kc":
            st.caption(
                "OpenET requires OPENET_API_KEY in Streamlit secrets. "
                "Use Hybrid for operational scheduling: OpenET supplies available past/recent actual ET, "
                "and Open-Meteo ET0 × Kc supplies forecast ET for future dates."
            )
            with st.expander("OpenET advanced settings"):
                openet_model = st.selectbox(
                    "OpenET model",
                    options=["Ensemble", "SSEBop", "eeMETRIC", "SIMS", "PT-JPL", "DisALEXI"],
                    index=0,
                )
                openet_interval = st.selectbox(
                    "OpenET interval",
                    options=["monthly", "daily"],
                    index=0,
                    help="Monthly is usually more reliable and is distributed evenly across days for the daily water balance. Use daily only if your OpenET access returns daily values.",
                )

    uploaded_csv = None
    if climate_source == "Upload daily climate CSV":
        st.markdown(
            """
            **CSV format requirements**  
            - Columns: `date`, `precip_mm`, `et0_mm`, `tmax_c`, `tmin_c`  
            - `date` in a format recognized by pandas (e.g., YYYY-MM-DD)
            """
        )
        uploaded_csv = st.file_uploader(
            "Upload daily climate CSV",
            type=["csv"],
        )

    if climate_source == "Simple ET₀ pattern (demo)":
        st.markdown("Simple sinusoidal ET₀ pattern with random rainfall for teaching / demos.")
        base_eto = st.slider("Base ET₀ (mm/day)", 3.0, 7.0, 5.0, 0.1)
        seasonal_amp = st.slider("Seasonal ET₀ amplitude (mm/day)", 0.0, 3.0, 1.0, 0.1)
        rain_prob = st.slider("Daily rainfall probability", 0.0, 0.7, 0.30, 0.05)
        mean_rain = st.slider("Mean rainfall on rainy days (mm)", 2.0, 25.0, 10.0, 0.5)
    else:
        base_eto = 5.0
        seasonal_amp = 1.0
        rain_prob = 0.30
        mean_rain = 10.0

    st.header("4. Irrigation system & strategy")

    irrigation_system = st.selectbox(
        "Irrigation system",
        options=list(IRRIGATION_SYSTEMS.keys()),
        index=0,
    )
    default_eff = IRRIGATION_SYSTEMS[irrigation_system]

    strategy_label = st.selectbox(
        "Irrigation strategy",
        options=list(STRATEGIES.keys()),
        index=0,
        help="Full irrigation keeps soil water high; deficit irrigation allows more stress.",
    )

    irrigation_eff = st.slider(
        "Irrigation application efficiency",
        min_value=0.6,
        max_value=0.95,
        value=float(default_eff),
        step=0.01,
        help="Can be adjusted around typical values for the selected system.",
    )
    rainfall_eff = st.slider(
        "Rainfall effectiveness",
        min_value=0.5,
        max_value=1.0,
        value=0.8,
        step=0.05,
    )


    default_event_depth_mm = {
        "Center pivot": 25.0,
        "Sprinkler (solid set/line)": 25.0,
        "Surface / Furrow": 80.0,
        "Drip": 10.0,
    }.get(irrigation_system, 25.0)

    max_event_depth_mm = {
        "Center pivot": 50.0,
        "Sprinkler (solid set/line)": 50.0,
        "Surface / Furrow": 150.0,
        "Drip": 25.0,
    }.get(irrigation_system, 50.0)

    irrigation_application_mm = st.slider(
        "Gross irrigation amount per event (mm)",
        min_value=5.0,
        max_value=max_event_depth_mm,
        value=default_event_depth_mm,
        step=5.0,
        key=f"event_depth_{irrigation_system}",
        help="Kansas center-pivot/sprinkler irrigation commonly applies about 25 mm per event. Surface/furrow events are usually larger.",
    )

    run_button = st.button("Run irrigation simulation", type="primary")

# TOP SECTION: MAP & SUMMARY CARDS
st.markdown('<span class="section-header">1. Kansas overview & field location</span>', unsafe_allow_html=True)

col_map, col_desc = st.columns([1.3, 1.2])

with col_map:
    df_map = pd.DataFrame({"lat": [lat], "lon": [lon]})
    st.map(df_map, zoom=6)

with col_desc:
    st.markdown(
        f"""
        <div class="metric-card">
        <b>Selected location:</b> {loc_name}<br>
        <b>Coordinates:</b> {lat:.3f}°N, {lon:.3f}°E<br>
        <b>Crop:</b> {crop_name}<br>
        <b>Soil (manual):</b> {soil_name}<br>
        <b>Irrigation system:</b> {irrigation_system}<br>
        <b>Strategy:</b> {strategy_label}<br>
        <b>ET method:</b> {et_method if climate_source == "Open-Meteo (automatic)" else climate_source}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if use_ssurgo:
        st.markdown(
            """
            <div class="metric-card">
            <b>SSURGO soil (beta):</b><br>
            If lookup was successful, dominant component and taxorder are shown in the sidebar.
            Use this info to refine your soil texture selection.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        This prototype focuses on **ET-based irrigation scheduling** for key crops in Kansas.  
        Use it to explore how **location, climate source, soil, irrigation system, and strategy** affect water use and yield.
        """
    )

st.markdown("---")

tabs = st.tabs(
    [
        "2. Weather & ET summary",
        "3. Scenario results: irrigation, ET deficit, and scheduling",
        "4. Detailed irrigation schedule & time series",
    ]
)

if run_button:
    crop = CROP_PARAMS[crop_name]
    season_len = crop["season_length_days"]

    with st.spinner("Preparing climate data and running irrigation simulation..."):
        if climate_source == "Open-Meteo (automatic)":
            df_weather = get_season_weather(lat, lon, planting_date, season_len)
        elif climate_source == "Simple ET₀ pattern (demo)":
            df_weather = generate_simple_eto_weather(
                planting_date,
                season_length=season_len,
                base_eto_mm=base_eto,
                seasonal_amp_mm=seasonal_amp,
                mean_precip_mm=mean_rain,
                rain_probability=rain_prob,
            )
        else:  # Upload daily climate CSV
            if uploaded_csv is None:
                st.error("Please upload a climate CSV file or choose another climate option.")
                df_weather = pd.DataFrame()
            else:
                df_weather = load_climate_from_csv(uploaded_csv, planting_date, season_len)

        if df_weather is None or df_weather.empty:
            st.error("No weather / climate data available for this configuration.")
        else:
            if climate_source == "Open-Meteo (automatic)":
                df_weather = apply_et_method(
                    df_weather=df_weather,
                    lat=lat,
                    lon=lon,
                    planting_date=planting_date,
                    season_length=season_len,
                    crop_name=crop_name,
                    et_method=et_method,
                    openet_model=openet_model,
                    openet_interval=openet_interval,
                )
            else:
                crop_kc = CROP_PARAMS[crop_name]["kc"]
                df_weather["openmeteo_etc_mm"] = df_weather["et0_mm"] * crop_kc
                df_weather["et_for_irrigation_mm"] = df_weather["openmeteo_etc_mm"]
                df_weather["et_source"] = "User/demo ET0 x Kc"

            irr_df, summary, df_weather = simulate_irrigation(
                df_weather,
                crop_name,
                soil_name,
                strategy_label,
                irrigation_system=irrigation_system,
                irrigation_efficiency=irrigation_eff,
                rainfall_efficiency=rainfall_eff,
                irrigation_application_mm=irrigation_application_mm,
            )

            # TAB 1: Weather & ET summary
            with tabs[0]:
                st.markdown('<span class="section-header">2. Weather, ET source, and seasonal water demand</span>', unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric(
                        "Season length (days)",
                        f"{len(df_weather):.0f}",
                    )
                with c2:
                    st.metric(
                        "Total precipitation (mm)",
                        f"{df_weather['precip_mm'].sum():.1f}",
                    )
                with c3:
                    st.metric(
                        "Total ET used by model (mm)",
                        f"{df_weather['etc_mm'].sum():.1f}",
                        help="This is ET0 x Kc for Open-Meteo/demo/CSV, or OpenET actual ET where selected and available.",
                    )
                with c4:
                    st.metric(
                        "Mean Tmax / Tmin (°C)",
                        f"{df_weather['tmax_c'].mean():.1f} / {df_weather['tmin_c'].mean():.1f}",
                    )

                chart_cols = ["time", "precip_mm", "et0_mm", "etc_mm"]
                if "openet_actual_et_mm" in df_weather.columns:
                    chart_cols.append("openet_actual_et_mm")
                chart_df = df_weather[chart_cols].copy()
                rename_map = {
                    "time": "Date",
                    "precip_mm": "Rain (mm)",
                    "et0_mm": "Reference ET0 (mm)",
                    "etc_mm": "ET used by model (mm)",
                    "openet_actual_et_mm": "OpenET actual ET (mm)",
                }
                chart_df = chart_df.rename(columns=rename_map)
                chart_df = chart_df.melt("Date", var_name="Variable", value_name="mm")

                st.line_chart(chart_df, x="Date", y="mm", color="Variable")

                if "et_source" in df_weather.columns:
                    st.markdown("#### ET source coverage")
                    st.caption("If OpenET is not configured or unavailable, this table will correctly show the Open-Meteo fallback source.")
                    source_table = (
                        df_weather.groupby("et_source", dropna=False)
                        .agg(days=("time", "count"), et_mm=("etc_mm", "sum"))
                        .reset_index()
                        .rename(columns={"et_source": "ET source", "et_mm": "ET used (mm)"})
                    )
                    show_small_table(source_table)

                st.caption(
                    "Open-Meteo provides reference ET0; the dashboard converts it to crop ET using Kc. "
                    "OpenET provides satellite-based actual ET and is used directly when selected."
                )

            # TAB 2: Scenario results summary
            with tabs[1]:
                st.markdown('<span class="section-header">3. Irrigation water use and ET deficit</span>', unsafe_allow_html=True)

                if irr_df is None or irr_df.empty:
                    st.info(
                        "No irrigations were triggered with the current assumptions. "
                        "This may occur if rainfall and ET are low, or if the season is very short."
                    )

                colA, colB, colC = st.columns(3)
                with colA:
                    st.metric(
                        "Total irrigation (mm)",
                        f"{summary['total_irrigation_mm']:.1f}",
                    )
                with colB:
                    st.metric(
                        "Number of irrigation events",
                        f"{summary['n_irrigations']}",
                    )
                with colC:
                    st.metric(
                        "Avg. irrigation/event (mm)",
                        f"{summary['avg_irrigation_event_mm']:.1f}",
                    )

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Total crop ET demand (mm)",
                        f"{summary['total_etc_mm']:.1f}",
                    )
                with col2:
                    st.metric(
                        "Cumulative ET deficit (mm)",
                        f"{summary['total_deficit_mm']:.1f}",
                    )

                summary_table = pd.DataFrame(
                    {
                        "Crop": [crop_name],
                        "Soil (manual)": [soil_name],
                        "Irrigation system": [irrigation_system],
                        "Strategy": [strategy_label],
                        "ET method": [et_method if climate_source == "Open-Meteo (automatic)" else climate_source],
                        "Root-zone TAW (mm)": [summary["taw_mm"]],
                        "Management depth (cm)": [summary["management_depth_m"] * 100.0],
                        "Top 30 cm available water (mm)": [summary["management_taw_mm"]],
                        "Irrigation trigger deficit (%)": [summary["trigger_depletion_pct"]],
                        "Irrigation trigger storage (%)": [summary["trigger_storage_pct"]],
                        "Trigger deficit in top 30 cm (mm)": [summary["trigger_depletion_mm"]],
                        "Total irrigation (mm)": [summary["total_irrigation_mm"]],
                        "Irrigation events (#)": [summary["n_irrigations"]],
                        "Total ETc (mm)": [summary["total_etc_mm"]],
                        "Cum. deficit (mm)": [summary["total_deficit_mm"]],
                        "Avg. irrigation/event (mm)": [summary["avg_irrigation_event_mm"]],
                        "ET demand met (%)": [summary["et_demand_met_pct"]],
                    }
                )
                st.markdown("#### Scenario summary table")
                show_small_table(summary_table.round(2))

                st.markdown(
                    f"""
                    - **Root-zone TAW** is retained for seasonal water-balance summaries.
                    - **Automatic irrigation trigger** is now based on the **top 30 cm management depth**, following the DSSAT-style 50% threshold.
                    - **Irrigation trigger used:** {summary['trigger_rule']}
                    - Use this tab to compare strategies, irrigation systems, and climate scenarios.
                    """
                )

            # TAB 3: Detailed schedule & time series
            with tabs[2]:
                st.markdown('<span class="section-header">4. Daily soil water balance and irrigation schedule</span>', unsafe_allow_html=True)

                if irr_df is None or irr_df.empty:
                    st.info("No irrigation events were triggered in this simulation.")
                else:
                    st.success(
                        f"{len(irr_df)} irrigation events were triggered. "
                        "The detailed event-date table has been hidden to keep this page clean."
                    )

                water_view = st.radio(
                    "Soil water variable to display",
                    options=[
                        "Deficit (% of top 30 cm available water)",
                        "Soil storage (% of top 30 cm available water)",
                    ],
                    index=0,
                    horizontal=True,
                    key="soil_water_variable_to_display",
                    help=(
                        "Choose one soil-water variable. Both are shown as percent of the "
                        "maximum available water in the top 30 cm management depth."
                    ),
                )

                if water_view.startswith("Deficit"):
                    water_col = "management_deficit_pct"
                    water_label = "Soil water deficit (% of top 30 cm available water)"
                    trigger_label = "50% deficit trigger"
                    trigger_value = float(summary.get("trigger_depletion_pct", 50.0))
                    trigger_caption = (
                        "Irrigation starts only when the deficit reaches the 50% trigger line "
                        "for the top 30 cm management depth."
                    )
                    y_domain = [0, 100]
                else:
                    water_col = "management_storage_pct"
                    water_label = "Soil water storage (% of top 30 cm available water)"
                    trigger_label = "50% storage trigger"
                    trigger_value = float(summary.get("trigger_storage_pct", 50.0))
                    trigger_caption = (
                        "Irrigation starts only when storage falls to the 50% trigger line "
                        "for the top 30 cm management depth."
                    )
                    y_domain = [0, 100]

                required_ts_cols = ["time", water_col, "irrigation_applied_mm"]
                missing_ts_cols = [c for c in required_ts_cols if c not in df_weather.columns]

                st.markdown("##### Irrigation applied and soil water status")
                if missing_ts_cols or df_weather.empty:
                    st.warning(
                        "Soil water balance columns are unavailable, so the time-series chart was skipped. "
                        f"Missing columns: {missing_ts_cols}"
                    )
                else:
                    ts = df_weather[required_ts_cols].copy()
                    ts["Date"] = pd.to_datetime(ts["time"], errors="coerce")
                    ts["water_value"] = pd.to_numeric(ts[water_col], errors="coerce").clip(0, 100)
                    ts["irrigation_applied_mm"] = pd.to_numeric(
                        ts["irrigation_applied_mm"], errors="coerce"
                    ).fillna(0.0)
                    ts["trigger_value"] = trigger_value
                    ts = ts.dropna(subset=["Date", "water_value"]).copy()

                    water_line = (
                        alt.Chart(ts)
                        .mark_line(point=False)
                        .encode(
                            x=alt.X("Date:T", title="Date"),
                            y=alt.Y(
                                "water_value:Q",
                                title=water_label,
                                scale=alt.Scale(domain=y_domain),
                            ),
                            tooltip=[
                                alt.Tooltip("Date:T", title="Date"),
                                alt.Tooltip("water_value:Q", title=water_label, format=".1f"),
                            ],
                        )
                    )

                    trigger_rule_line = (
                        alt.Chart(ts)
                        .mark_rule(strokeDash=[6, 4])
                        .encode(
                            y=alt.Y(
                                "trigger_value:Q",
                                title=water_label,
                                scale=alt.Scale(domain=y_domain),
                            ),
                            tooltip=[
                                alt.Tooltip("trigger_value:Q", title=trigger_label, format=".1f"),
                            ],
                        )
                    )

                    irrigation_bars = (
                        alt.Chart(ts)
                        .mark_bar(opacity=0.55)
                        .encode(
                            x=alt.X("Date:T", title="Date"),
                            y=alt.Y(
                                "irrigation_applied_mm:Q",
                                title="Irrigation applied (mm)",
                            ),
                            tooltip=[
                                alt.Tooltip("Date:T", title="Date"),
                                alt.Tooltip(
                                    "irrigation_applied_mm:Q",
                                    title="Irrigation applied (mm)",
                                    format=".1f",
                                ),
                            ],
                        )
                    )

                    # Use two vertically aligned panels instead of a dual-axis layer.
                    # This avoids the earlier soil-storage display problem and makes
                    # both deficit and storage options work reliably.
                    water_chart = alt.layer(water_line, trigger_rule_line).properties(height=300)
                    irrigation_chart = irrigation_bars.properties(height=120)
                    chart = alt.vconcat(water_chart, irrigation_chart).resolve_scale(x="shared")
                    st.altair_chart(chart, use_container_width=True)

                    st.caption(
                        "The upper panel shows the selected soil-water status as percent of maximum "
                        "available water in the top 30 cm management depth. The lower panel shows "
                        "gross irrigation applied in mm. "
                        f"{trigger_caption}"
                    )

                st.caption(
                    "Values are conceptual and for demonstration only. For operational scheduling, "
                    "this framework should be calibrated with local soil and crop data from Kansas fields."
                )

else:
    with tabs[0]:
        st.info("Set your location, crop, climate option, soil, and irrigation system in the sidebar, then click **Run irrigation simulation**.")
    with tabs[1]:
        st.info("Scenario results will appear here after running a simulation.")
    with tabs[2]:
        st.info("Daily irrigation schedule and time series will be shown after running a simulation.")