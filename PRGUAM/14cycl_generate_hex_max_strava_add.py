#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hexová agregace cyklistiky – zjednodušená verze pro data Strava_cycl_sum_year.gpkg,
kde již je celkový počet jízd (obousměrně) v poli 'total_trip_count'
a není třeba zohledňovat typ cesty ani rychlosti.

Postup:
1. načte hexy (EPSG:5514) a linie (typicky EPSG:4326);
2. přepočte hodnoty na denní (pokud PERIOD='annual');
3. provede overlay linek s hexy (intersection);
4. pro každý hex spočte MAX z _count_daily;
5. výsledek uloží jako <PREFIX>_cyclist_trip_count_in_hex.

Výstupní vrstva: hexy s atributem <PREFIX>_cyclist_trip_count_in_hex.
"""

import sys
import pandas as pd
import geopandas as gpd

# ========================
# ======== KONFIG ========
# ========================

HEX_GPKG    = r"G:\Python_env_PragUAM\testing_location_01\h3_res12_testing_location_01_cyclists_unimotion_daily.gpkg"
OUTPUT_GPKG = r"G:\Python_env_PragUAM\testing_location_01\h3_res12_testing_location_01_cyclists_combined_daily.gpkg"


# HEX_GPKG   = r"G:\Python_env_PragUAM\cycl_strava_hex\h3_res12_praha_100m_cyclists_unimotion_daily.gpkg"  # hexy (EPSG:5514)
# OUTPUT_GPKG = r"G:\Python_env_PragUAM\cycl_strava_hex\h3_res12_praha_100m_cyclists_combined_daily.gpkg"


# HEX_GPKG    = r"G:\Python_env_PragUAM\testing_location_03\h3_res12_testing_location_03_cyclists_unimotion_daily.gpkg"
# OUTPUT_GPKG = r"G:\Python_env_PragUAM\testing_location_03\h3_res12_testing_location_03_cyclists_combined_daily.gpkg"


LINES_GPKG  = r"G:\Python_env_PragUAM\Strava_cycl_sum_year.gpkg"
HEX_ID_FIELD   = "h3_id"
COUNT_FIELD    = "total_trip_count"

PERIOD = "annual"          # "daily" nebo "annual"
ANNUAL_DENOMINATOR = 365.0 # pokud PERIOD="annual", dělí se tímto číslem

ATTR_PREFIX = "strava"
TARGET_CRS  = "EPSG:5514"

# ========================


def log(msg):
    print(f"\033[96m[INFO]\033[0m {msg}", flush=True)


def list_layers_gpkg(path):
    layers = None
    try:
        import fiona
        layers = fiona.listlayers(path)
    except Exception:
        pass
    if not layers:
        try:
            from pyogrio import list_layers
            layers = [name for name, _ in list_layers(path)]
        except Exception:
            pass
    if not layers:
        raise RuntimeError(f"Nelze vypsat vrstvy v GPKG: {path}.")
    return layers


def read_first_layer(path):
    layers = list_layers_gpkg(path)
    log(f"Nalezeno {len(layers)} vrstev v '{path}': {', '.join(layers)}")
    if len(layers) > 1:
        print(f"  -> Beru první: '{layers[0]}'", file=sys.stderr)
    
    layer_name = layers[0]
    gdf = gpd.read_file(path, layer=layer_name)
    return gdf, layer_name

def ensure_crs(gdf: gpd.GeoDataFrame, target: str, label: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise RuntimeError(f"Vrstva '{label}' nemá CRS.")
    if str(gdf.crs).lower() != target.lower():
        log(f"Reprojekce '{label}' z {gdf.crs} do {target}")
        return gdf.to_crs(target)
    log(f"Vrstva '{label}' je již v {target}")
    return gdf


def main():
    log("=== ZAČÁTEK ===")

    # 1) načtení
    log(f"Načítám hexy: {HEX_GPKG}")
    hexy, hex_layer_name = read_first_layer(HEX_GPKG)

    log(f"Načítám linie: {LINES_GPKG}")
    lines, _ = read_first_layer(LINES_GPKG)

    # 2) hex ID
    if HEX_ID_FIELD not in hexy.columns:
        log(f"Pole '{HEX_ID_FIELD}' nenalezeno – vytvořím z indexu.")
        hexy = hexy.copy()
        hexy[HEX_ID_FIELD] = hexy.index.astype(str)

    # 3) CRS sjednocení
    hexy  = ensure_crs(hexy, TARGET_CRS, "hexy")
    lines = ensure_crs(lines, TARGET_CRS, "linie")

    # 4) zkontroluj sloupec s počtem
    if COUNT_FIELD not in lines.columns:
        raise RuntimeError(f"Chybí pole '{COUNT_FIELD}' ve vrstv̌e linií.")

    # 5) přepočet na denní
    val = pd.to_numeric(lines[COUNT_FIELD], errors="coerce").fillna(0.0)
    if PERIOD.lower() == "annual":
        denom = ANNUAL_DENOMINATOR if ANNUAL_DENOMINATOR > 0 else 365.0
        lines["_count_daily"] = val / denom
        log(f"Roční vstup – dělím {denom}.")
    else:
        lines["_count_daily"] = val
        log("Denní vstup – beze změny.")

    # 6) overlay linek s hexy
    log("Provádím overlay (intersection)...")
    seg = gpd.overlay(lines, hexy[[HEX_ID_FIELD, "geometry"]], how="intersection", keep_geom_type=True)
    log(f"Segmentů po overlay: {len(seg)}")
    if seg.empty:
        log("Průnik prázdný – ukládám hexy s nulami.")
        out_field = f"{ATTR_PREFIX}_cyclist_trip_count_in_hex"
        hexy[out_field] = 0.0
        hexy.to_file(OUTPUT_GPKG, layer="hexy_out", driver="GPKG")
        log("=== KONEC ===")
        return

    # 7) spočti MAX z denních hodnot v každém hexu
    out_field = f"{ATTR_PREFIX}_cyclist_trip_count_in_hex"
    log("Počítám MAX(_count_daily) pro každý hex...")
    agg = seg.groupby(HEX_ID_FIELD)["_count_daily"].max().rename(out_field).reset_index()

    # 8) merge do hexů
    hexy_out = hexy.merge(agg, on=HEX_ID_FIELD, how="left")
    hexy_out[out_field] = hexy_out[out_field].fillna(0.0)

    # 9) uložení
    log(f"Ukládám výsledek do {OUTPUT_GPKG} (vrstva: {hex_layer_name})")
    # Pozn.: stejné jméno vrstvy -> vrstva se přepíše novou verzí s přidaným sloupcem
    hexy_out.to_file(OUTPUT_GPKG, layer=hex_layer_name, driver="GPKG")
    log(f"Hotovo. Přidaný sloupec: '{out_field}'")
    log("=== KONEC ===")


if __name__ == "__main__":
    main()
