# -*- coding: utf-8 -*-
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from fiona import listlayers

# =========================
# INPUTS (EDIT)
# =========================


# Hex grid (must contain column "h3")
HEX_GPKG  = r"G:\Python_env_PragUAM\testing_location_01\h3_res12_testing_location_01_100m_5514.gpkg"
OUT_GPKG = r"G:\Python_env_PragUAM\testing_location_01\ppl_in_hex_res12_testing_location_01_100m_5514.gpkg"



HEX_LAYER = None  # pokud je v gpkg jedna vrstva, nech None

# UNION_FINAL (ZSJ × landuse × budovy)
UNION_GPKG  = r"G:\Python_env_PragUAM\ppl_golemio\FINALNI_UNION_VRSTEV\UNION_FINAL.gpkg"
UNION_LAYER = "UNION_FINAL"


H3_COL    = "h3"

# Dasymetric settings
ZSJ_KEY_COL     = "ZSJ_Kod_ZSJ"
POP_COL         = "ZSJ_Celkem_avg"
LANDUSE_CODE_COL= "VYUZ_KOD"

# capacity from buildings (preferred)
HPP_COL         = "PODL_HPP_prepocet"   # kapacita dílku, po rozdělení budov

# CSV mappings
CODES_TO_GROUP_CSV = r"G:\Python_env_PragUAM\ppl_golemio\landuse_codes_to_group.csv"
GROUP_WEIGHTS_CSV  = r"G:\Python_env_PragUAM\ppl_golemio\landuse_group_weights.csv"
PROFILE            = "day"  # day | workday | weekend | night

# capacity rule:
#  - "hpp_or_area": capacity = HPP_prepocet pokud >0, jinak area (umožní i “silnice” atd. dle vah)
#  - "hpp_only": jen HPP (ne-budovy dostanou 0 => pokud je chceš úplně vyloučit)
#  - "area_only": jen plocha (bez budov, jen landuse)
CAPACITY_MODE = "hpp_or_area"

# OUTPUT
OUT_LAYER = "ppl_in_hex"

# =========================
# Helpers
# =========================

def _read_gpkg(path, layer=None):
    if layer is None:
        layers = listlayers(path)
        layer = layers[0]
    return gpd.read_file(path, layer=layer)

def _ensure_h3_as_str(gdf: gpd.GeoDataFrame, h3_col: str) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    if h3_col not in gdf.columns:
        raise ValueError(f"Missing required H3 column '{h3_col}'")
    gdf[h3_col] = gdf[h3_col].astype(str)
    return gdf

def _load_codes_to_group(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"code", "group"}.issubset(df.columns):
        raise ValueError("codes_to_group musí mít sloupce: code, group")
    df = df.copy()
    df["code"] = df["code"].astype(str)
    return df[["code", "group"]]

def _load_group_weights(path: str, profile: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "group" not in df.columns or profile not in df.columns:
        raise ValueError(f"group_weights musí mít sloupce: group a {profile}")
    return df[["group", profile]].rename(columns={profile: "weight"})

# =========================
# 1) Load data
# =========================

union = gpd.read_file(UNION_GPKG, layer=UNION_LAYER)
hexy  = _read_gpkg(HEX_GPKG, HEX_LAYER)

if union.crs is None or hexy.crs is None:
    raise ValueError("Jedna z vrstev nemá definované CRS.")
if union.crs != hexy.crs:
    hexy = hexy.to_crs(union.crs)

# geometry fix (optional)
union = union.copy()
union["geometry"] = union.geometry.buffer(0)
hexy = hexy.copy()
hexy["geometry"] = hexy.geometry.buffer(0)



# H3
hexy = _ensure_h3_as_str(hexy, H3_COL)

# checks
for c in [ZSJ_KEY_COL, POP_COL, LANDUSE_CODE_COL]:
    if c not in union.columns:
        raise KeyError(f"UNION chybí sloupec '{c}'")

if CAPACITY_MODE != "area_only" and HPP_COL not in union.columns:
    raise KeyError(f"UNION chybí sloupec HPP '{HPP_COL}', ale CAPACITY_MODE={CAPACITY_MODE}")

# =========================
# 2) Dasymetric allocation on UNION pieces
# =========================

union["_area_m2"] = union.geometry.area

codes = _load_codes_to_group(CODES_TO_GROUP_CSV)
weights = _load_group_weights(GROUP_WEIGHTS_CSV, PROFILE)

union["_code"] = union[LANDUSE_CODE_COL].astype(str)
# ošetření chybějících land-use kódů
union["_code"] = union["VYUZ_KOD"].astype("string").fillna("UNKNOWN")

union = union.merge(codes.rename(columns={"code": "_code"}), on="_code", how="left")
if union["group"].isna().any():
    missing = sorted(union.loc[union["group"].isna(), "_code"].unique().tolist())
    raise ValueError(f"Chybí mapování code->group pro: {missing[:50]}")

union = union.merge(weights, on="group", how="left")
if union["weight"].isna().any():
    bad = sorted(union.loc[union["weight"].isna(), "group"].unique().tolist())
    raise ValueError(f"Chybí váhy pro skupiny: {bad}")

# capacity
if CAPACITY_MODE == "area_only":
    union["_capacity"] = union["_area_m2"].astype(float)
elif CAPACITY_MODE == "hpp_only":
    hpp = pd.to_numeric(union[HPP_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    union["_capacity"] = hpp
else:  # hpp_or_area
    hpp = pd.to_numeric(union[HPP_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    union["_capacity"] = np.where(hpp.to_numpy() > 0.0, hpp.to_numpy(), union["_area_m2"].to_numpy())

# base and allocation within each ZSJ
union["_base"] = union["weight"].to_numpy() * union["_capacity"].to_numpy()
base_sum = union.groupby(ZSJ_KEY_COL)["_base"].sum().rename("_base_sum")
union = union.join(base_sum, on=ZSJ_KEY_COL)

bad_zsj = union.loc[np.isclose(union["_base_sum"], 0.0), ZSJ_KEY_COL].unique()
if len(bad_zsj) > 0:
    raise RuntimeError(f"ZSJ s nulovým součtem vah/kapacit (ukázka): {bad_zsj[:20]}")

union["ppl_alloc"] = union[POP_COL].astype(float) * (union["_base"] / union["_base_sum"])

# keep only what we need for overlay to hex
union_alloc = union[["ppl_alloc", "_area_m2", "geometry"]].copy()

# =========================
# 3) Transfer ppl_alloc to hexes by area ratio
# =========================

# Spatial join for candidate pairs
pairs = gpd.sjoin(
    hexy[[H3_COL, "geometry"]],
    union_alloc,
    predicate="intersects",
    how="left"
)

# pairs now has geometry = hex geometry; union geometry is accessible via index_right
pairs = pairs.dropna(subset=["index_right"]).copy()
if len(pairs) == 0:
    # no intersections
    out = hexy.copy()
    out["sum_ppl_in_hex"] = 0.0
    out["hex_density_km2"] = 0.0
else:
    # fetch union geoms + values
    u = union_alloc.loc[pairs["index_right"].astype(int)].reset_index(drop=True)
    h = hexy.loc[pairs.index].reset_index(drop=True)

    inter_geom = u.geometry.intersection(h.geometry)
    inter_area = inter_geom.area.to_numpy(dtype=float)
    union_area = u["_area_m2"].to_numpy(dtype=float)

    ratio = np.divide(inter_area, union_area, out=np.zeros_like(inter_area), where=union_area > 0.0)
    ppl_part = u["ppl_alloc"].to_numpy(dtype=float) * ratio

    tmp = pd.DataFrame({H3_COL: h[H3_COL].astype(str).to_numpy(), "ppl_part": ppl_part})
    ppl_hex = tmp.groupby(H3_COL, as_index=False)["ppl_part"].sum().rename(columns={"ppl_part": "sum_ppl_in_hex"})

    # join back to all hexes (keep full grid)
    out = hexy.merge(ppl_hex, on=H3_COL, how="left")
    out["sum_ppl_in_hex"] = out["sum_ppl_in_hex"].fillna(0.0)

    # density (optional)
    out["_hex_area_m2"] = out.geometry.area
    out["hex_density_km2"] = out["sum_ppl_in_hex"] / (out["_hex_area_m2"] / 1e6).replace(0, np.nan)
    out["hex_density_km2"] = out["hex_density_km2"].fillna(0.0)
    out = out.drop(columns=["_hex_area_m2"], errors="ignore")

# =========================
# 4) Save
# =========================
Path(OUT_GPKG).parent.mkdir(parents=True, exist_ok=True)
out.to_file(OUT_GPKG, layer=OUT_LAYER, driver="GPKG")

print("=== DONE: ppl_in_hex (dasymetric + HPP) ===")
print(f"Written: {OUT_GPKG} | layer: {OUT_LAYER}")
print(f"Hex count: {len(out)}")
print(f"Total population sum: {out['sum_ppl_in_hex'].sum():,.0f}")
