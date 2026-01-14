#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hexová agregace cyklistiky – MAX průjezdů + délky s >0 + průměrný okamžitý počet (proměnná rychlost dle roadType).
Vše nastavíš v sekci KONFIG níže. Skript píše průběžné INFO logy.
Počítá se bez délkového vážení průjezdu (žádné len_seg/len_orig).

Výstupní pole v hexech:
  <PREFIX>_cyclist_trip_count_in_hex  ... maximum denních jízd v hexu
  <PREFIX>_tracks_length_in_hex       ... součet délek (m) segmentů s denním průjezdem > 0 (duplicitní směry -> délka 2×)
  <PREFIX>_cyclist_in_hex             ... průměrný okamžitý počet cyklistů v hexu

Pozn.: Pokud máš dopravní úseky jako dvě shodné geometrie (každý směr zvlášť),
      overlay je rozdělí a výše uvedené se chová správně (délky se sčítají 2×, MAX z denních jízd vezme větší z obou směrů).
"""

import sys
import math
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely import set_precision

# =========================
# ======== KONFIG =========
# =========================

# --- Cesty k datům ---

HEX_GPKG   = r"G:\Python_env_PragUAM\testing_location_01\testing_location_01_h3_shelter_obstacle.gpkg"    # hexy (EPSG:5514)
OUTPUT_GPKG = r"G:\Python_env_PragUAM\testing_location_01\h3_res12_testing_location_01_cyclists_unimotion_daily.gpkg"


# HEX_GPKG   = r"G:\Python_env_PragUAM\DSM_DTM\h3_res12_polygons_praha_100m_5514.gpkg"   # hexy (EPSG:5514)                           # linky (typicky EPSG:4326)
# OUTPUT_GPKG = r"G:\Python_env_PragUAM\cycl_strava_hex\h3_res12_praha_100m_cyclists_unimotion_daily.gpkg"

# HEX_GPKG   = r"G:\Python_env_PragUAM\testing_location_03\h3_res12_polygons_testing_location_03_5514.gpkg"    # hexy (EPSG:5514)            
# OUTPUT_GPKG = r"G:\Python_env_PragUAM\testing_location_03\h3_res12_testing_location_03_cyclists_unimotion_daily.gpkg"

LINES_GPKG = r"G:\Python_env_PragUAM\cyclist_unimotion.gpkg"     # linky (typicky EPSG:4326)
# --- Sloupce / názvy ---
HEX_ID_FIELD   = "h3_id"           # pole s ID hexu (vytvoří se z indexu, pokud neexistuje)
COUNT_FIELD    = "2024_year_everyday_estimated_daily_counts" # denní/roční počet jízd NA FUNKCI (každý směr má vlastní featuru se stejným názvem pole)
ROADTYPE_FIELD = "roadType"        # pole s typem komunikace (pro mapu rychlostí níže)

# --- Periodicita vstupních hodnot ---
PERIOD = "daily"   # "daily" nebo "annual"
ANNUAL_DENOMINATOR = 365.0

# --- Okno aktivních hodin (pro průměrný okamžitý počet) ---
ACTIVE_START_H = 6   # 0–23
ACTIVE_END_H   = 22  # 0–23  (6–22 => 16 h = 57600 s)

# --- Rychlosti (km/h) podle roadType + DEFAULT ---
SPEED_BY_ROADTYPE_KMH = {
    # přizpůsob si – příklady:
    "CYCLEWAY":      16.0,
    "BRIDLEWAY":     12.0,
    "FOOTWAY":        6.0,
    "PATH":          10.0,
    "CROSSING":       5.0,
    "LIVING_STREET": 12.0,
    "PRIMARY":       18.0,
    "SECONDARY":     18.0,
    "TERTIARY":      16.0,
    "QUATERNARY":    14.0,
    "STAIRS":         3.0,
    "TRACK":         12.0,
    # atd...
}
DEFAULT_SPEED_KMH = 10.0  # když roadType není ve slovníku nebo je NULL

# --- Prefix výstupních atributů ---
ATTR_PREFIX = "unimotion"

# --- Cílové CRS ---
TARGET_CRS = "EPSG:5514"

# =========================
# ====== /KONFIG ==========
# =========================

# pro stabilní porovnání geometrií (metrový CRS → 0.01 m mřížka obvykle stačí)
PRECISION_GRID = 0.01

# =========================

def log(msg: str):
    print(f"\033[96m[INFO]\033[0m {msg}", flush=True)

def list_layers_gpkg(path: str):
    layers = None
    try:
        import fiona
        layers = fiona.listlayers(path)
    except Exception:
        layers = None
    if not layers:
        try:
            from pyogrio import list_layers
            layers = [name for name, _ in list_layers(path)]
        except Exception:
            layers = None
    if not layers:
        raise RuntimeError(f"Nelze vypsat vrstvy v GPKG: {path}. Nainstaluj 'fiona' nebo 'pyogrio'.")
    return layers

def read_first_layer(path: str) -> gpd.GeoDataFrame:
    layers = list_layers_gpkg(path)
    log(f"Nalezeno {len(layers)} vrstev v '{path}': {', '.join(layers)}")
    if len(layers) > 1:
        print(f"  -> Beru první: '{layers[0]}'", file=sys.stderr)
    return gpd.read_file(path, layer=layers[0])

def ensure_crs(gdf: gpd.GeoDataFrame, target: str, label: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise RuntimeError(f"Vrstva '{label}' nemá definované CRS.")
    if str(gdf.crs).lower() != target.lower():
        log(f"Reprojekce '{label}' z {gdf.crs} do {target}")
        return gdf.to_crs(target)
    log(f"Vrstva '{label}' je již v {target}")
    return gdf

def canonicalize_linestring(ls: LineString) -> LineString:
    # odstraň vliv směru: vyber lexikograficky "menší" konec jako začátek
    coords = list(ls.coords)
    if coords[0] > coords[-1]:
        coords = coords[::-1]
    return LineString(coords)

def edge_key_from_geom(geom) -> str:
    """
    Vytvoří klíč hrany nezávislý na směru a s nastavenou přesností.
    Pokud jsou dvě featury opačným směrem po stejné čáře, dostanou stejný key.
    """
    g = set_precision(geom, PRECISION_GRID)
    if isinstance(g, LineString):
        g2 = canonicalize_linestring(g)
        return g2.wkb_hex
    elif isinstance(g, MultiLineString):
        parts = [canonicalize_linestring(ls) for ls in g.geoms]
        # seřaď podle prvního bodu pro stabilitu
        parts_sorted = sorted(parts, key=lambda p: (p.coords[0][0], p.coords[0][1], len(p.coords)))
        merged = MultiLineString(parts_sorted)
        return merged.wkb_hex
    else:
        # pro jistotu – na liniové vrstvě by nastat nemělo
        return set_precision(g, PRECISION_GRID).wkb_hex

def speed_mps(roadtype) -> float:
    if pd.isna(roadtype):
        kmh = DEFAULT_SPEED_KMH
    else:
        kmh = SPEED_BY_ROADTYPE_KMH.get(str(roadtype).upper(), DEFAULT_SPEED_KMH)
    return max(0.001, kmh * 1000.0 / 3600.0)

def main():
    log("=== ZAČÁTEK ===")
    log(f"Načítám hexy: {HEX_GPKG}")
    hexy = read_first_layer(HEX_GPKG)
    log(f"Načítám linie: {LINES_GPKG}")
    lines = read_first_layer(LINES_GPKG)

    if HEX_ID_FIELD not in hexy.columns:
        log(f"Pole '{HEX_ID_FIELD}' v hexech nenalezeno – vytvořím z indexu.")
        hexy = hexy.copy()
        hexy[HEX_ID_FIELD] = hexy.index.astype(str)

    hexy  = ensure_crs(hexy, TARGET_CRS, "hexy")
    lines = ensure_crs(lines, TARGET_CRS, "linie")

    if COUNT_FIELD not in lines.columns:
        raise RuntimeError(f"Ve vrstv̌e linií chybí '{COUNT_FIELD}'.")
    if ROADTYPE_FIELD not in lines.columns:
        log(f"Upozornění: '{ROADTYPE_FIELD}' chybí – použiji default rychlost {DEFAULT_SPEED_KMH} km/h pro všechny.")

    # denní hodnota
    val = pd.to_numeric(lines[COUNT_FIELD], errors="coerce").fillna(0.0)
    if PERIOD.lower() == "annual":
        denom = ANNUAL_DENOMINATOR if ANNUAL_DENOMINATOR and ANNUAL_DENOMINATOR > 0 else 365.0
        lines["_count_daily"] = (val / float(denom)).astype(float)
        log(f"Vstup je roční – dělím {denom} → _count_daily.")
    elif PERIOD.lower() == "daily":
        lines["_count_daily"] = val.astype(float)
        log("Vstup je denní – _count_daily = COUNT_FIELD.")
    else:
        raise RuntimeError("PERIOD musí být 'daily' nebo 'annual'.")

    # degeneráty pryč
    lines["len_orig"] = lines.geometry.length
    lines = lines[lines["len_orig"] > 0].copy()
    log(f"Po odfiltrování nulové délky: {len(lines)} linií")

    # edge_key dle geometrie (nezávislý na směru)
    log("Vytvářím 'edge_key' ze znormalizované geometrie (nezávislý na směru)...")
    lines["edge_key"] = lines.geometry.apply(edge_key_from_geom)

    # overlay: linie ∩ hexy → segmenty
    log("Provádím overlay (intersection) ...")
    seg = gpd.overlay(lines, hexy[[HEX_ID_FIELD, "geometry"]], how="intersection", keep_geom_type=True)
    log(f"Segmentů po overlay: {len(seg)}")
    if seg.empty:
        log("Průnik prázdný – ukládám hexy s nulami.")
        hexy[f"{ATTR_PREFIX}_cyclist_trip_count_in_hex"] = 0.0
        hexy[f"{ATTR_PREFIX}_tracks_length_in_hex"] = 0.0
        hexy[f"{ATTR_PREFIX}_cyclist_in_hex"] = 0.0
        hexy.to_file(OUTPUT_GPKG, layer="hexy_out", driver="GPKG")
        log("=== KONEC ===")
        return

    seg["len_seg"] = seg.geometry.length
    seg = seg[seg["len_seg"] > 0].copy()
    log(f"Segmenty s nenulovou délkou: {len(seg)}")

    # rychlost m/s podle roadType
    if ROADTYPE_FIELD in seg.columns:
        seg["_speed_mps"] = seg[ROADTYPE_FIELD].apply(speed_mps)
    else:
        seg["_speed_mps"] = DEFAULT_SPEED_KMH * 1000.0 / 3600.0

    # === KLÍČOVÁ OPRAVA ===
    # 1) v každém hexu nejdřív sečti směry TÉŽE hrany (podle edge_key)
    #    → dostaneme 'edge_daily_sum' per (hex, edge_key)
    per_hex_edge = (
        seg.groupby([HEX_ID_FIELD, "edge_key"])["_count_daily"]
        .sum()
        .reset_index(name="edge_daily_sum")
    )

    # 2) MAX přes hrany v hexu (tj. max ze součtu směrů)
    out_max_name = f"{ATTR_PREFIX}_cyclist_trip_count_in_hex"
    max_trips = (
        per_hex_edge.groupby(HEX_ID_FIELD)["edge_daily_sum"]
        .max()
        .rename(out_max_name)
    )

    # 3) Součet délek všech segmentů v hexu s _count_daily > 0
    out_len_name = f"{ATTR_PREFIX}_tracks_length_in_hex"
    seg_pos = seg[seg["_count_daily"] > 0]
    sum_len = seg_pos.groupby(HEX_ID_FIELD)["len_seg"].sum().rename(out_len_name)

    # 4) Suma časů na 1 průjezd hexem (∑ len/speed) – každý směr zvlášť
    seg_pos = seg_pos.copy()
    seg_pos["time_per_trip_s"] = seg_pos["len_seg"] / seg_pos["_speed_mps"]
    sum_time_one_trip = seg_pos.groupby(HEX_ID_FIELD)["time_per_trip_s"].sum()

    # aktivní okno
    if not (0 <= ACTIVE_START_H <= 23 and 0 <= ACTIVE_END_H <= 23):
        raise RuntimeError("ACTIVE_START_H a ACTIVE_END_H musí být v rozsahu 0–23.")
    active_hours = (ACTIVE_END_H - ACTIVE_START_H) % 24
    if active_hours == 0:
        raise RuntimeError("Délka aktivního okna vychází 0 h – uprav ACTIVE_*.")
    active_secs = active_hours * 3600.0
    log(f"Aktivní okno: {active_hours} h = {int(active_secs)} s")

    # průměrný okamžitý počet v hexu
    out_avg_name  = f"{ATTR_PREFIX}_cyclist_in_hex"
    out_time_name = f"{ATTR_PREFIX}_time_per_trip_in_hex_s"
    agg = pd.concat(
        [max_trips, sum_len, sum_time_one_trip.rename(out_time_name)],
        axis=1
    ).fillna(0.0).reset_index()

    def _avg(row):
        trips = float(row[out_max_name])
        t_one = float(row[out_time_name])
        if trips <= 0.0 or t_one <= 0.0:
            return 0.0
        return (trips * t_one) / active_secs

    agg[out_avg_name] = agg.apply(_avg, axis=1)

    # výstup
    hexy_out = hexy.merge(agg, on=HEX_ID_FIELD, how="left")
    for c in [out_max_name, out_len_name, out_avg_name, out_time_name]:
        hexy_out[c] = pd.to_numeric(hexy_out[c], errors="coerce").fillna(0.0)

    log(f"Ukládám výsledek do {OUTPUT_GPKG}")
    hexy_out.to_file(OUTPUT_GPKG, layer="hexy_out", driver="GPKG")
    log("Hotovo.")
    log(f"  - {out_max_name} (MAX ze součtu směrů)")
    log(f"  - {out_len_name} (m; směr >0 → počítá se; při dvou směrech >0 je délka 2×)")
    log(f"  - {out_time_name} (s; suma časů jednoho průjezdu hexem)")
    log(f"  - {out_avg_name} (průměrný okamžitý počet)")
    log("=== KONEC ===")

if __name__ == "__main__":
    main()