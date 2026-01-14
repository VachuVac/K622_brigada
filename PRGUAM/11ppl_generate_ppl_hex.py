# -*- coding: utf-8 -*-
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

# ==========
# INPUTS
# ==========
ZSJ_GPKG = r"G:\Python_env_PragUAM\ppl_golemio\ppl_zsj_stats_5514.gpkg"
ZSJ_LAYER = "ppl_zsj_stats_en"

HEX_GPKG = r"G:\Python_env_PragUAM\testing_location_01\h3_res12_testing_location_01_100m_5514.gpkg" 
HEX_LAYER = None  # pokud je v GPKG jen jedna vrstva

OUT_GPKG = r"G:\Python_env_PragUAM\testing_location_01\OLD_ppl_in_hex_res12_testing_location_01_100m_5514.gpkg"
OUT_LAYER = "ppl_in_hex"

# ZSJ_GPKG = r"G:\Python_env_PragUAM\ppl_golemio\ppl_zsj_stats_5514.gpkg"
# ZSJ_LAYER = "ppl_zsj_stats_en"

# HEX_GPKG = r"G:\Python_env_PragUAM\testing_location_03\h3_res12_polygons_testing_location_03_5514.gpkg" 
# HEX_LAYER = None  # pokud je v GPKG jen jedna vrstva

# OUT_GPKG = r"G:\Python_env_PragUAM\testing_location_03\ppl_in_hex_res12_testing_location_03_100m_5514.gpkg"
# OUT_LAYER = "ppl_in_hex"
# ==========
# 1) LOAD DATA
# ==========
zsj = gpd.read_file(ZSJ_GPKG, layer=ZSJ_LAYER)
hexy = gpd.read_file(HEX_GPKG, layer=HEX_LAYER)

# kontrola CRS a sjednocení
if zsj.crs is None or hexy.crs is None:
    raise ValueError("Jedna z vrstev nemá definované CRS.")
if zsj.crs != hexy.crs:
    zsj = zsj.to_crs(hexy.crs)

print(f"ZSJ loaded: {len(zsj)} polygons | CRS: {zsj.crs}")
print(f"Hexes loaded: {len(hexy)} | CRS: {hexy.crs}")

# ==========
# 2) PREPARE DATA
# ==========
# najdi sloupec s ID hexu
hex_id_col = None
for c in hexy.columns:
    if "h3" in c.lower() and "id" in c.lower():
        hex_id_col = c
        break
if hex_id_col is None:
    # fallback: první ne-geometrický sloupec
    hex_id_col = [c for c in hexy.columns if c != "geometry"][0]

# kontrola hustoty
if "zsj_density_km2" not in zsj.columns:
    raise KeyError("Ve vrstvě ZSJ chybí sloupec 'zsj_density_km2'.")

# ==========
# 3) INTERSECTION – jen v rozsahu hexů
# ==========
print("Provádím prostorový průnik (intersection)...")

inter = gpd.overlay(
    zsj[["zsj_code", "zsj_density_km2", "geometry"]],
    hexy[[hex_id_col, "geometry"]],
    how="intersection"
)

print(f"Počet průniků: {len(inter)}")

# ==========
# 4) AREA A POPULATION PER INTERSECTION
# ==========
inter["hex_part_area_m2"] = inter.geometry.area
inter["hex_part_area_km2"] = inter["hex_part_area_m2"] / 1_000_000.0

# počet osob v dané části hexu
inter["ppl_in_hex_part"] = inter["zsj_density_km2"] * inter["hex_part_area_km2"]

# ==========
# 5) AGGREGACE PO HEXECH
# ==========
ppl_hex = (
    inter.groupby(hex_id_col)
         .agg(
             sum_ppl_in_hex=("ppl_in_hex_part", "sum"),
             hex_area_m2=("hex_part_area_m2", "sum"),
             src_zsj_codes=("zsj_code", lambda vals: ",".join(sorted(set(map(str, vals))))),
             src_density_mean_km2=("zsj_density_km2", "mean"),
         )
         .reset_index()
)

ppl_hex["hex_area_km2"] = ppl_hex["hex_area_m2"] / 1_000_000.0
ppl_hex["hex_density_km2"] = ppl_hex.apply(
    lambda r: r["sum_ppl_in_hex"] / r["hex_area_km2"] if r["hex_area_km2"] > 0 else 0, axis=1
)

# ==========
# 6) PŘIPOJ GEOMETRII HEXŮ
# ==========
ppl_hex = hexy[[hex_id_col, "geometry"]].merge(ppl_hex, on=hex_id_col, how="inner")
ppl_hex = gpd.GeoDataFrame(ppl_hex, geometry="geometry", crs=hexy.crs)

# ==========
# 7) SAVE
# ==========
Path(OUT_GPKG).parent.mkdir(parents=True, exist_ok=True)
ppl_hex.to_file(OUT_GPKG, layer=OUT_LAYER, driver="GPKG")

# ==========
# 8) SUMMARY
# ==========
print("=== DONE: ppl_in_hex ===")
print(f"Written: {OUT_GPKG} | layer: {OUT_LAYER}")
print(f"Hex count: {len(ppl_hex)}")
print(f"Total population sum: {ppl_hex['sum_ppl_in_hex'].sum():,.0f}")
print
