import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from fiona import listlayers

# ---------------------------
# User inputs (edit as needed)
# ---------------------------
# BASE hex grid with full attribute table to start from
BASE_HEX_GPKG = r"G:\Python_env_PragUAM\testing_location_01\h3_res12_testing_location_01_100m_5514.gpkg"
PPL_GPKG = r"G:\Python_env_PragUAM\testing_location_01\OLD_ppl_in_hex_res12_testing_location_01_100m_5514.gpkg"
CYCL_GPKG = r"G:\Python_env_PragUAM\testing_location_01\h3_res12_testing_location_01_cyclists_combined_daily_total.gpkg"
PED_GPKG = r"G:\Python_env_PragUAM\testing_location_01\h3_res12_location_01_pedestrians_daily_MAX.gpkg"
SHELTER_GPKG = r"G:\Python_env_PragUAM\testing_location_01\testing_location_01_h3_shelter_flight_p50.gpkg"
OUT_GPKG = r"G:\Python_env_PragUAM\testing_location_01\OLD_testing_location_01_h3_shelter_factor_results.gpkg"


# BASE_HEX_GPKG = r"G:\Python_env_PragUAM\testing_location_03\h3_res12_polygons_testing_location_03_5514.gpkg"
# PPL_GPKG = r"G:\Python_env_PragUAM\testing_location_03\ppl_in_hex_res12_testing_location_03_100m_5514.gpkg"
# CYCL_GPKG = r"G:\Python_env_PragUAM\testing_location_03\h3_res12_testing_location_03_cyclists_combined_daily_total.gpkg"
# PED_GPKG = r"G:\Python_env_PragUAM\testing_location_03\h3_res12_location_03_pedestrians_daily_MAX.gpkg"
# SHELTER_GPKG = r"G:\Python_env_PragUAM\testing_location_03\testing_location_03_h3_shelter_flight_p50.gpkg"
# OUT_GPKG = r"G:\Python_env_PragUAM\testing_location_03\testing_location_03_h3_shelter_factor_results.gpkg"



BASE_HEX_LAYER = None
PPL_LAYER = None
PPL_COL = "sum_ppl_in_hex"
CYCL_LAYER = None
CYCL_COL = "total_cyclist_in_hex"
PED_LAYER = None
PED_COL = "ped_ped_in_hex"
SHELTER_LAYER = None
SHELTER_COL_REL = "shelter_obstacle_rel"
SHELTER_COL_GLIDE = "glide_shelter_prob"
SHELTER_COL_BALL = "ballistic_shelter_prob"  # "parabolic" v zadání
H3_COL = "h3"
OUT_LAYER = "h3_shelter_factor_results"

# ---------------------------
# Helpers
# ---------------------------

def _read_gpkg(path, layer=None):
    """Read a (layer from) GPKG. If layer is None and multiple layers exist, read the first."""
    if layer is None:
        try:
            layers = listlayers(path)
            layer = layers[0]
        except Exception:
            layer = None
    return gpd.read_file(path, layer=layer)

def _read_layer_with_columns(path: str, required_cols: list[str]) -> gpd.GeoDataFrame:
    """Najde první vrstvu v GPKG, která obsahuje všechny požadované sloupce."""
    layers = listlayers(path)
    last_err = None
    for lyr in layers:
        try:
            gdf = gpd.read_file(path, layer=lyr, rows=1)
            if all(c in gdf.columns for c in required_cols):
                print(f"✔ Using layer '{lyr}' from {path}")
                return gpd.read_file(path, layer=lyr)
        except Exception as e:
            last_err = e
            continue
    raise ValueError(
        f"No layer in '{path}' contains required columns {required_cols}. "
        f"Last error: {last_err}"
    )

def _ensure_h3_as_str(gdf: gpd.GeoDataFrame, h3_col: str) -> gpd.GeoDataFrame:
    """Cast H3 column to string to guarantee consistent joins."""
    gdf = gdf.copy()
    if h3_col not in gdf.columns:
        raise ValueError(f"Missing required H3 column '{h3_col}' in input GeoDataFrame")
    gdf[h3_col] = gdf[h3_col].astype(str)
    return gdf

def join_metric_on_h3(base_gdf: gpd.GeoDataFrame,
                      metric_gdf: gpd.GeoDataFrame,
                      h3_col: str,
                      metric_col: str,
                      out_col: str) -> gpd.GeoDataFrame:
    """Left-join jedné metriky přes společný H3 klíč."""
    base_gdf = _ensure_h3_as_str(base_gdf, h3_col)
    metric_gdf = _ensure_h3_as_str(metric_gdf, h3_col)

    # Když metrika chybí ve zdroji, přidej nuly
    if metric_col not in metric_gdf.columns:
        rhs = metric_gdf[[h3_col]].drop_duplicates().copy()
        rhs[metric_col] = 0.0
    else:
        rhs = metric_gdf[[h3_col, metric_col]].copy()

    same_name = (out_col == metric_col)
    tmp_col = f"__tmp__{metric_col}" if same_name else metric_col
    if same_name:
        rhs = rhs.rename(columns={metric_col: tmp_col})

    merged = base_gdf.merge(rhs, how='left', on=h3_col)

    src_col = tmp_col if same_name else metric_col
    merged[out_col] = pd.to_numeric(merged[src_col], errors='coerce').fillna(0.0)

    if src_col in merged.columns and src_col != out_col:
        merged = merged.drop(columns=[src_col])

    return merged

def compute_igrc(n_unsheltered: pd.Series, cap_to_10: bool = True) -> pd.Series:
    """
    JARUS SORA Annex F 2.5:
      iGRC = max(1, ceil(7 + log10(N) - 0.5)), ale pro N <= 0 -> 0
    - N = počet 'unsheltered people' v hexu pro daný režim
    - vrací int (0..10, pokud cap_to_10=True)
    """
    n = pd.to_numeric(n_unsheltered, errors="coerce").fillna(0.0)
    out = pd.Series(np.zeros(len(n), dtype=np.int64), index=n.index)  # default 0 pro N <= 0

    pos = n > 0
    if pos.any():
        vals = 7.0 + np.log10(n[pos])
        igrc = np.ceil(vals - 0.5).astype(int)
        igrc = np.maximum(igrc, 1)  # u kladných N min. 1
        if cap_to_10:
            igrc = np.minimum(igrc, 10)  # horní strop podle běžné škály iGRC
        out.loc[pos] = igrc

    return out.astype(int)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Read BASE hex grid (we keep all its attributes and geometry)
    base = _read_gpkg(BASE_HEX_GPKG, BASE_HEX_LAYER)

    # Validate H3 in base
    if H3_COL not in base.columns:
        raise ValueError(f"Base hex grid is missing H3 column '{H3_COL}' -> {BASE_HEX_GPKG}")
    base = _ensure_h3_as_str(base, H3_COL)

    # Read other layers
    ppl = _read_layer_with_columns(PPL_GPKG, [H3_COL, PPL_COL]) if PPL_LAYER is None else _read_gpkg(PPL_GPKG, PPL_LAYER)
    cycl = _read_layer_with_columns(CYCL_GPKG, [H3_COL, CYCL_COL]) if CYCL_LAYER is None else _read_gpkg(CYCL_GPKG, CYCL_LAYER)
    ped  = _read_layer_with_columns(PED_GPKG,  [H3_COL, PED_COL])  if PED_LAYER  is None else _read_gpkg(PED_GPKG,  PED_LAYER)
    shel = _read_layer_with_columns(SHELTER_GPKG, [H3_COL, SHELTER_COL_REL, SHELTER_COL_GLIDE, SHELTER_COL_BALL]) if SHELTER_LAYER is None else _read_gpkg(SHELTER_GPKG, SHELTER_LAYER)

    # Normalize H3 as string
    for g in (ppl, cycl, ped, shel):
        _ = _ensure_h3_as_str(g, H3_COL)

    # Diagnostics
    print("--- Diagnostics (H3 coverage) ---")
    print("Base hexes:", len(base))
    print("PPL unique H3:", ppl[H3_COL].nunique(), "metric:", PPL_COL in ppl.columns)
    print("CYCL unique H3:", cycl[H3_COL].nunique(), "metric:", CYCL_COL in cycl.columns)
    print("PED unique H3:", ped[H3_COL].nunique(), "metric:", PED_COL in ped.columns)
    print("SHEL unique H3:", shel[H3_COL].nunique(),
          "metrics present:", all(c in shel.columns for c in [SHELTER_COL_REL, SHELTER_COL_GLIDE, SHELTER_COL_BALL]))

    # Attach metrics by H3 join
    base = join_metric_on_h3(base, ppl,  H3_COL, PPL_COL,  'sum_ppl_in_hex')
    base = join_metric_on_h3(base, cycl, H3_COL, CYCL_COL, 'total_cyclist_in_hex')
    base = join_metric_on_h3(base, ped,  H3_COL, PED_COL,  'ped_ped_in_hex')
    # Shelter probabilities
    base = join_metric_on_h3(base, shel, H3_COL, SHELTER_COL_REL,  SHELTER_COL_REL)
    base = join_metric_on_h3(base, shel, H3_COL, SHELTER_COL_GLIDE, SHELTER_COL_GLIDE)
    base = join_metric_on_h3(base, shel, H3_COL, SHELTER_COL_BALL,  SHELTER_COL_BALL)

    # Ensure numeric
    for col in ['sum_ppl_in_hex', 'total_cyclist_in_hex', 'ped_ped_in_hex']:
        if col not in base.columns:
            base[col] = 0.0
        base[col] = pd.to_numeric(base[col], errors='coerce').fillna(0.0)

    # Total heads
    base['total_heads_in_hex'] = (
        base['sum_ppl_in_hex'].fillna(0.0)
        + base['total_cyclist_in_hex'].fillna(0.0)
        + base['ped_ped_in_hex'].fillna(0.0)
    )

    # Probabilities
    shelter_rel = pd.to_numeric(base[SHELTER_COL_REL],  errors='coerce').fillna(0.0).clip(0, 1)
    p_glide     = pd.to_numeric(base[SHELTER_COL_GLIDE], errors='coerce').fillna(0.0).clip(0, 1)
    p_ball      = pd.to_numeric(base[SHELTER_COL_BALL],  errors='coerce').fillna(0.0).clip(0, 1)

    p_comb_glide = 1 - (1 - shelter_rel) * (1 - p_glide)
    p_comb_ball  = 1 - (1 - shelter_rel) * (1 - p_ball)

    # Unsheltered headcounts by regime
    base['unsheltered_total_raw']       = base['total_heads_in_hex'] * (1 - shelter_rel)   # obstacles-only
    base['unsheltered_total_glide']     = base['total_heads_in_hex'] * (1 - p_comb_glide)  # glide combined
    base['unsheltered_total_ballistic'] = base['total_heads_in_hex'] * (1 - p_comb_ball)   # parabolic/ballistic combined

    # Shares (optional)
    denom = base['total_heads_in_hex'].replace(0, np.nan)
    base['unsheltered_total_raw_share']       = (base['unsheltered_total_raw']       / denom).fillna(0.0)
    base['unsheltered_total_glide_share']     = (base['unsheltered_total_glide']     / denom).fillna(0.0)
    base['unsheltered_total_ballistic_share'] = (base['unsheltered_total_ballistic'] / denom).fillna(0.0)

    # ---------------------------
    # iGRC per regime (requested)
    # ---------------------------
    base['iGRC_obstacles'] = compute_igrc(base['unsheltered_total_raw'])
    base['iGRC_glide']     = compute_igrc(base['unsheltered_total_glide'])
    base['iGRC_parabolic'] = compute_igrc(base['unsheltered_total_ballistic'])  # alias “parabolic”

    # Housekeeping
    for c in ['unsheltered_total_raw', 'unsheltered_total_glide',
              'unsheltered_total_ballistic', 'total_heads_in_hex']:
        base[c] = base[c].fillna(0.0)

    # Save output
    Path(OUT_GPKG).parent.mkdir(parents=True, exist_ok=True)
    base.to_file(OUT_GPKG, layer=OUT_LAYER, driver='GPKG')
    print(f"Saved: {OUT_GPKG} -> layer '{OUT_LAYER}'")
