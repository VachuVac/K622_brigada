#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agregace chodců (Strava pedestrians) do H3 hexů:
 - overlay linií s hexy,
 - denní průměr z ročního počtu,
 - délky tras s nenulovým průchodem,
 - výpočet průměrného okamžitého počtu chodců v hexu.
"""

import geopandas as gpd
import pandas as pd
import math

# ==========================
# ======== KONFIG ==========
# ==========================
HEX_GPKG   = r"G:\Python_env_PragUAM\testing_location_01\testing_location_01_h3_shelter_obstacle.gpkg" 
OUTPUT_GPKG = r"G:\Python_env_PragUAM\testing_location_01\h3_res12_location_01_pedestrians_daily_MAX.gpkg"

# HEX_GPKG   = r"G:\Python_env_PragUAM\testing_location_03\h3_res12_polygons_testing_location_03_5514.gpkg" 
# OUTPUT_GPKG = r"G:\Python_env_PragUAM\testing_location_03\h3_res12_location_03_pedestrians_daily_MAX.gpkg"
LINES_GPKG = r"G:\Python_env_PragUAM\strava_ped_sum_year_2024_5514.gpkg"

HEX_ID_FIELD  = "h3_id"
COUNT_FIELD   = "total_trip_count"     # roční počet průchodů
PERIOD        = "annual"               # nebo "daily"
ANNUAL_DENOMINATOR = 365.0

# Rychlost chodce (km/h)
SPEED_KMH = 3.0
# Aktivní okno (hodiny) – 2–22
ACTIVE_START_H = 2
ACTIVE_END_H   = 22

# Prefix pro výstupní atributy
ATTR_PREFIX = "ped"

# ==========================


def log(msg):
    print(f"\033[96m[INFO]\033[0m {msg}", flush=True)


def list_layers_gpkg(path):
    try:
        import fiona
        return fiona.listlayers(path)
    except Exception:
        try:
            from pyogrio import list_layers
            return [n for n, _ in list_layers(path)]
        except Exception:
            return None


def read_first_layer(path):
    layers = list_layers_gpkg(path)
    if not layers:
        raise RuntimeError(f"Nelze vypsat vrstvy v GPKG: {path}")
    if len(layers) > 1:
        log(f"Upozornění: více vrstev ({len(layers)}), beru první: {layers[0]}")
    return gpd.read_file(path, layer=layers[0])


def main():
    log("=== ZAČÁTEK ===")

    log(f"Načítám hexy: {HEX_GPKG}")
    hexy = read_first_layer(HEX_GPKG)
    log(f"Načítám linie chodců: {LINES_GPKG}")
    lines = read_first_layer(LINES_GPKG)

    # Ověření pole ID
    if HEX_ID_FIELD not in hexy.columns:
        hexy[HEX_ID_FIELD] = hexy.index.astype(str)
        log(f"Přidáno pole '{HEX_ID_FIELD}' z indexu.")

    # Reprojekce (obě 5514)
    if hexy.crs != lines.crs:
        lines = lines.to_crs(hexy.crs)
        log("Reprojekce linií na CRS hexů.")

    # Zkontroluj atribut
    if COUNT_FIELD not in lines.columns:
        raise RuntimeError(f"Chybí atribut {COUNT_FIELD} ve vrstvě linií!")

    # Denní hodnota
    val = pd.to_numeric(lines[COUNT_FIELD], errors="coerce").fillna(0.0)
    if PERIOD == "annual":
        lines["_count_daily"] = val / ANNUAL_DENOMINATOR
        log("Roční vstup – dělím 365 → denní průměr.")
    else:
        lines["_count_daily"] = val
        log("Denní vstup – beze změny.")

    # Délky linií
    lines["len_orig"] = lines.geometry.length
    lines = lines[lines["len_orig"] > 0].copy()
    log(f"Počet linií s délkou > 0: {len(lines)}")

    # Overlay (intersection)
    log("Provádím průnik linií a hexů ...")
    seg = gpd.overlay(lines, hexy[[HEX_ID_FIELD, "geometry"]], how="intersection", keep_geom_type=True)
    if seg.empty:
        log("Výsledek prázdný – končím.")
        return
    seg["len_seg"] = seg.geometry.length
    seg = seg[seg["len_seg"] > 0].copy()
    log(f"Počet segmentů s délkou > 0: {len(seg)}")

    # Segmenty s nenulovým průchodem
    seg_pos = seg[seg["_count_daily"] > 0].copy()
    log(f"Segmentů s průchodem > 0: {len(seg_pos)}")

    # Rychlost m/s
    speed_mps = SPEED_KMH * 1000.0 / 3600.0

    # Výpočet času pro 1 průchod segmentem
    seg_pos["time_per_trip_s"] = seg_pos["len_seg"] / speed_mps

    # Aktivní doba v sekundách
    active_hours = (ACTIVE_END_H - ACTIVE_START_H) % 24
    if active_hours == 0:
        active_hours = 20  # fallback
    active_secs = active_hours * 3600.0
    log(f"Aktivní okno: {active_hours} h = {int(active_secs)} s")

    # Agregace do hexů
    out_max = f"{ATTR_PREFIX}_trip_count_in_hex"
    out_len = f"{ATTR_PREFIX}_tracks_length_in_hex"
    out_avg = f"{ATTR_PREFIX}_ped_in_hex"

    max_trips = seg.groupby(HEX_ID_FIELD)["_count_daily"].max().rename(out_max)
    sum_len = seg_pos.groupby(HEX_ID_FIELD)["len_seg"].sum().rename(out_len)
    sum_time_one_trip = seg_pos.groupby(HEX_ID_FIELD)["time_per_trip_s"].sum()

    agg = pd.concat([max_trips, sum_len, sum_time_one_trip.rename("_sum_time_one_trip_s")], axis=1).fillna(0.0).reset_index()

    def _avg(row):
        trips = row[out_max]
        t_one = row["_sum_time_one_trip_s"]
        if trips <= 0.0 or t_one <= 0.0:
            return 0.0
        return (trips * t_one) / active_secs

    agg[out_avg] = agg.apply(_avg, axis=1)
    agg = agg.drop(columns=["_sum_time_one_trip_s"])

    # Join zpět na hexy
    hexy_out = hexy.merge(agg, on=HEX_ID_FIELD, how="left")
    for c in [out_max, out_len, out_avg]:
        hexy_out[c] = pd.to_numeric(hexy_out[c], errors="coerce").fillna(0.0)

    log(f"Ukládám výsledek do {OUTPUT_GPKG}")
    hexy_out.to_file(OUTPUT_GPKG, layer="hexy_out", driver="GPKG")

    log("=== HOTOVO ===")
    log(f"Pole: {out_max}, {out_len}, {out_avg}")


if __name__ == "__main__":
    main()
