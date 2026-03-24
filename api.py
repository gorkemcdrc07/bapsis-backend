from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import tempfile
import os
import traceback

from truck_routing_v18 import TariffEngine

app = FastAPI(title="Truck Routing API")

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://YOUR-FRONTEND-PROJECT.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def safe_float(v, default=0.0):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v, default=0):
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def dataframe_to_records(df: pd.DataFrame):
    if df is None or df.empty:
        return []

    clean_df = df.copy()

    for col in clean_df.columns:
        if pd.api.types.is_float_dtype(clean_df[col]):
            clean_df[col] = clean_df[col].fillna(0).astype(float)
        elif pd.api.types.is_integer_dtype(clean_df[col]):
            clean_df[col] = clean_df[col].fillna(0).astype(int)
        else:
            clean_df[col] = clean_df[col].where(pd.notnull(clean_df[col]), None)

    return clean_df.to_dict(orient="records")


def build_totals(summary_df: pd.DataFrame, truck_capacity: float):
    if summary_df is None or summary_df.empty:
        return {
            "total_cost": 0,
            "total_pallets": 0,
            "avg_utilization_pct": 0,
            "fleet_utilization_pct": 0,
            "truck_count": 0,
        }

    total_cost = safe_float(summary_df["total_cost"].sum())
    total_pallets = safe_float(summary_df["total_pallets"].sum())
    avg_util = safe_float(summary_df["utilization_%"].mean())
    truck_count = len(summary_df)

    fleet_util = 0
    if truck_count > 0 and truck_capacity > 0:
        fleet_util = (total_pallets / (truck_count * truck_capacity)) * 100

    return {
        "total_cost": round(total_cost, 2),
        "total_pallets": round(total_pallets, 2),
        "avg_utilization_pct": round(avg_util, 1),
        "fleet_utilization_pct": round(fleet_util, 1),
        "truck_count": truck_count,
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/plan")
async def plan(
    data_file: UploadFile = File(...),
    demand_file: UploadFile = File(...),
    selected_date: str = Form(...),
    strategy: str = Form("greedy"),
    max_stops: int = Form(3),
    fixed_cost: float = Form(0),
):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, data_file.filename or "TruckRoutingData.xlsx")
            demand_path = os.path.join(tmpdir, demand_file.filename or "TruckRoutingDemands.xlsx")

            with open(data_path, "wb") as f:
                f.write(await data_file.read())

            with open(demand_path, "wb") as f:
                f.write(await demand_file.read())

            engine = TariffEngine(data_path, demand_path)
            engine.params["max_stops_per_truck"] = max_stops
            engine.params["vehicle_fixed_cost"] = fixed_cost

            if strategy == "ortools":
                plan_data = engine.optimize_with_ortools(selected_date)
            else:
                plan_data = engine.allocate_greedy(selected_date)

            if not plan_data:
                return {
                    "plan": {},
                    "summary": [],
                    "totals": {
                        "total_cost": 0,
                        "total_pallets": 0,
                        "avg_utilization_pct": 0,
                        "fleet_utilization_pct": 0,
                        "truck_count": 0,
                    },
                    "truck_capacity": safe_int(getattr(engine, "truck_capacity", 33), 33),
                    "depot_name": getattr(engine, "depot_name", ""),
                    "missing_tariffs": [],
                }

            summary = engine.multi_truck_costs(plan_data)

            if isinstance(summary, pd.DataFrame) and not summary.empty:
                summary = summary.sort_values("total_cost", ascending=False).reset_index(drop=True)
                totals = build_totals(summary, safe_float(engine.truck_capacity, 33))
                summary_records = dataframe_to_records(summary)
            else:
                totals = build_totals(pd.DataFrame(), safe_float(engine.truck_capacity, 33))
                summary_records = []

            missing_tariffs = sorted(list(getattr(engine, "missing_tariffs", set())))

            return {
                "plan": plan_data,
                "summary": summary_records,
                "totals": totals,
                "truck_capacity": safe_int(getattr(engine, "truck_capacity", 33), 33),
                "depot_name": getattr(engine, "depot_name", ""),
                "missing_tariffs": missing_tariffs,
            }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )