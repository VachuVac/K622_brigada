"""
shelter_3D_combined.py

Computes both linear ("glide") and parabolic ("ballistic") shelter factors
for all hex centroids and stores everything in one GeoPackage layer.
"""

import math
import numpy as np
import geopandas as gpd
from hex_shelter_calculation_v2 import compute_shelter_for_points_v2

# ---------------- CONFIG ----------------
GPKG_PATH = r"G:\Python_env_PragUAM\testing_location_01\testing_location_01_h3_shelter_obstacle.gpkg"
OUT_GPKG  = r"G:\Python_env_PragUAM\testing_location_01\testing_location_01_h3_shelter_flight_p50.gpkg"

# GPKG_PATH = r"G:\Python_env_PragUAM\testing_location_03\testing_location_03_h3_shelter_obstacle.gpkg"
# OUT_GPKG  = r"G:\Python_env_PragUAM\testing_location_03\testing_location_03_h3_shelter_flight_p50.gpkg"

LAYER_NAME = "shelter_3D_combined"
OUT_LAYER = "shelter_3D_flight"

# data columns
COL_ID  = "h3"
COL_DTM = "DTM_hex_p50"
COL_OBJ = "Rel_height_hex_p50"

# ---------------- SHARED PARAMETERS (same for both drones) ----------------
INITIAL_HEIGHT  = 100.0   # z0 (m)
PERSON_HEIGHT   = 1.8     # observer height (m)
THETA_DEG       = 0.0     # launch elevation angle for parabolic model (deg)

# ---------------- DRONE-SPECIFIC CONFIGS ----------------
# GLIDE (linear)
GLIDE_CFG = {
    "ANGLE_OF_ATTACK": 35.0,  # deg; only used by linear model
    # the following are ignored by the linear model, but kept for symmetry / future use
    "V0": 26.38889,           # m/s (ignored by linear)
    "DRAG_ENABLED": False,    # ignored by linear
    "RHO": 1.225,
    "CD": 0.8,
    "AREA": 0.5,
    "MASS": 6.8,
    "DT_INT": 0.02,
    "MAX_STEPS": 20000,
}

# BALLISTIC (parabolic)
BALLISTIC_CFG = {
    "V0": 16.0,          # m/s  
    "DRAG_ENABLED": True,
    "RHO": 1.225,
    "CD": 0.8,
    "AREA": 0.5,        # m^2
    "MASS": 13.2,         # kg
    "DT_INT": 0.01,
    "MAX_STEPS": 30000,
    # angle of attack not used; THETA_DEG is shared above
}

# ---------------- BEHAVIOR ----------------
MIN_NEIGHBORS_TO_COUNT = 1
VERBOSE_SHELTER = True
PROGRESS_EVERY  = 2000


def compute_cell_distance(x, y):
    """Median nearest-neighbor distance in meters."""
    coords = np.c_[x, y]
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=2)
        nearest = dists[:, 1]
        return float(np.median(nearest))
    except Exception:
        n = len(coords)
        nearest = np.full(n, np.inf, dtype=float)
        for i in range(n):
            dx = coords[i, 0] - coords[:, 0]
            dy = coords[i, 1] - coords[:, 1]
            d = np.hypot(dx, dy)
            d[i] = np.inf
            nearest[i] = d.min()
        return float(np.median(nearest))


def run_glide(gdf, cell_distance):
    """Run linear (glide) model with its own drone parameters."""
    print("\n--- Computing GLIDE (linear) ---")
    sp, ncount, depth, max_range, meta = compute_shelter_for_points_v2(
        x=gdf["cx"].to_numpy(dtype=float),
        y=gdf["cy"].to_numpy(dtype=float),
        dtm=gdf[COL_DTM].to_numpy(dtype=float),
        obj=np.nan_to_num(gdf[COL_OBJ].to_numpy(dtype=float), nan=0.0),
        angle_deg=GLIDE_CFG["ANGLE_OF_ATTACK"],
        z0=INITIAL_HEIGHT,
        person_h=PERSON_HEIGHT,
        traj_mode="linear",
        # parabolic params are ignored in linear mode, but we pass sensible defaults
        v0=GLIDE_CFG["V0"], theta_deg=THETA_DEG, h0=None,
        drag_enabled=GLIDE_CFG["DRAG_ENABLED"], rho=GLIDE_CFG["RHO"],
        Cd=GLIDE_CFG["CD"], area=GLIDE_CFG["AREA"], mass=GLIDE_CFG["MASS"],
        dt_int=GLIDE_CFG["DT_INT"], max_steps=GLIDE_CFG["MAX_STEPS"],
        cell_distance=cell_distance, min_neighbors_to_count=MIN_NEIGHBORS_TO_COUNT,
        widen_factor=1.5, verbose=VERBOSE_SHELTER, progress_every=PROGRESS_EVERY,
    )

    prefix = "glide_"
    gdf[f"{prefix}shelter_prob"]   = sp
    gdf[f"{prefix}neighbors"]      = ncount
    gdf[f"{prefix}depth"]          = depth
    gdf[f"{prefix}max_range_m"]    = max_range
    gdf[f"{prefix}cell_distance_m"]= cell_distance
    gdf[f"{prefix}angle_deg"]      = GLIDE_CFG["ANGLE_OF_ATTACK"]
    gdf[f"{prefix}z0_m"]           = INITIAL_HEIGHT
    gdf[f"{prefix}person_h_m"]     = PERSON_HEIGHT
    gdf[f"{prefix}traj_mode"]      = "linear"
    print("GLIDE done. Meta:", meta)
    return gdf


def run_ballistic(gdf, cell_distance):
    """Run parabolic (ballistic) model with its own drone parameters."""
    print("\n--- Computing BALLISTIC (parabolic) ---")
    sp, ncount, depth, max_range, meta = compute_shelter_for_points_v2(
        x=gdf["cx"].to_numpy(dtype=float),
        y=gdf["cy"].to_numpy(dtype=float),
        dtm=gdf[COL_DTM].to_numpy(dtype=float),
        obj=np.nan_to_num(gdf[COL_OBJ].to_numpy(dtype=float), nan=0.0),
        angle_deg=GLIDE_CFG["ANGLE_OF_ATTACK"],  # not used in parabolic calc
        z0=INITIAL_HEIGHT,
        person_h=PERSON_HEIGHT,
        traj_mode="parabolic",
        v0=BALLISTIC_CFG["V0"], theta_deg=THETA_DEG, h0=None,
        drag_enabled=BALLISTIC_CFG["DRAG_ENABLED"], rho=BALLISTIC_CFG["RHO"],
        Cd=BALLISTIC_CFG["CD"], area=BALLISTIC_CFG["AREA"], mass=BALLISTIC_CFG["MASS"],
        dt_int=BALLISTIC_CFG["DT_INT"], max_steps=BALLISTIC_CFG["MAX_STEPS"],
        cell_distance=cell_distance, min_neighbors_to_count=MIN_NEIGHBORS_TO_COUNT,
        widen_factor=1.5, verbose=VERBOSE_SHELTER, progress_every=PROGRESS_EVERY,
    )

    prefix = "ballistic_"
    gdf[f"{prefix}shelter_prob"]    = sp
    gdf[f"{prefix}neighbors"]       = ncount
    gdf[f"{prefix}depth"]           = depth
    gdf[f"{prefix}max_range_m"]     = max_range
    gdf[f"{prefix}cell_distance_m"] = cell_distance
    gdf[f"{prefix}angle_deg"]       = GLIDE_CFG["ANGLE_OF_ATTACK"]  # kept for symmetry
    gdf[f"{prefix}z0_m"]            = INITIAL_HEIGHT
    gdf[f"{prefix}person_h_m"]      = PERSON_HEIGHT
    gdf[f"{prefix}traj_mode"]       = "parabolic"
    print("BALLISTIC done. Meta:", meta)
    return gdf


def main():
    print(f"Loading: {GPKG_PATH} layer: {LAYER_NAME}")
    gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME)

    for c in (COL_ID, COL_DTM, COL_OBJ, "geometry"):
        if c not in gdf.columns and c != "geometry":
            raise ValueError(f"Missing column '{c}'.")

    if gdf.crs is None:
        raise ValueError("Layer has no CRS; expected metric CRS (e.g., EPSG:5514).")
    print("Layer CRS:", gdf.crs)

    # centroids
    gdf["centroid"] = gdf.geometry.centroid
    gdf["cx"] = gdf.centroid.x
    gdf["cy"] = gdf.centroid.y

    # neighborhood scale (shared)
    print("Estimating typical neighbor distance (m)…")
    cell_distance = compute_cell_distance(gdf["cx"], gdf["cy"])
    print("Cell distance (median NN):", round(cell_distance, 3), "m")

    # compute both drones
    gdf = run_glide(gdf, cell_distance=cell_distance)
    gdf = run_ballistic(gdf, cell_distance=cell_distance)

    # cleanup helper columns
    if "centroid" in gdf.columns:
        gdf.drop(columns=["centroid"], inplace=True)

    print(f"\nSaving combined output → {OUT_GPKG} (layer: {OUT_LAYER})")
    gdf.to_file(OUT_GPKG, layer=OUT_LAYER, driver="GPKG")
    print("All done!")


if __name__ == "__main__":
    main()