# === TOTALS: Strava + Unimotion ====================================
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona

# --- Nastavení (úprav podle potřeby) ---
INPUT_GPKG = r"G:\Python_env_PragUAM\testing_location_01\h3_res12_testing_location_01_cyclists_combined_daily.gpkg"


# INPUT_GPKG    = r"G:\Python_env_PragUAM\cycl_strava_hex\h3_res12_praha_100m_cyclists_combined_daily.gpkg"
# INPUT_GPKG = r"G:\Python_env_PragUAM\testing_location_03\h3_res12_testing_location_03_cyclists_combined_daily.gpkg"

INPUT_LAYER   = None  # None = vezme první vrstvu v souboru
OUTPUT_GPKG   = os.path.splitext(INPUT_GPKG)[0] + "_total.gpkg"  # nebo = INPUT_GPKG pro přepsání
OUTPUT_LAYER  = None  # None = stejné jméno vrstvy jako vstupní
MERGE_THRESHOLD = 0.8     # 0.20 = Strava >= Unimotion * 1.20 -> sečíst
ACTIVE_SECS     = 16 * 3600  # jmenovatel pro průměrný okamžitý počet

# --- Očekávané vstupní sloupce ---
COL_STRAVA     = "strava_cyclist_trip_count_in_hex"
COL_UNIMOT_TRP = "unimotion_cyclist_trip_count_in_hex"
COL_TIME_S     = "unimotion_time_per_trip_in_hex_s"
COL_LEN_M      = "unimotion_tracks_length_in_hex"   # informativní

# --- Výstupní sloupce ---
COL_TOTAL_TRP  = "total_cyclist_trip_count_in_hex"
COL_TOTAL_AVG  = "total_cyclist_in_hex"

def _list_layers_gpkg(path: str):
    with fiona.Env():
        return fiona.listlayers(path)

def _read_layer(path: str, layer: str | None):
    layers = _list_layers_gpkg(path)
    lyr = layer or layers[0]
    if lyr not in layers:
        raise ValueError(f"Vrstva '{lyr}' v '{path}' neexistuje. Dostupné: {', '.join(layers)}")
    gdf = gpd.read_file(path, layer=lyr)
    return gdf, lyr

def _ensure_numeric(gdf: gpd.GeoDataFrame, cols):
    for c in cols:
        if c in gdf.columns:
            gdf[c] = pd.to_numeric(gdf[c], errors="coerce").fillna(0.0)
        else:
            # pokud chybí, doplníme 0.0 (bezpečné chování)
            gdf[c] = 0.0
    return gdf

# --- Načtení vstupu ---
gdf, in_layer = _read_layer(INPUT_GPKG, INPUT_LAYER)
out_layer = OUTPUT_LAYER or in_layer

# --- Ošetření typů / NaN ---
gdf = _ensure_numeric(gdf, [COL_STRAVA, COL_UNIMOT_TRP, COL_TIME_S, COL_LEN_M])

# --- MERGE logika: když je Strava o MERGE_THRESHOLD relativně vyšší než Unimotion, sečteme; jinak bereme Unimotion ---
cond = gdf[COL_STRAVA] >= gdf[COL_UNIMOT_TRP] * (1.0 + float(MERGE_THRESHOLD))
gdf[COL_TOTAL_TRP] = np.where(cond, gdf[COL_STRAVA] + gdf[COL_UNIMOT_TRP], gdf[COL_UNIMOT_TRP])

# --- Průměrný okamžitý počet cyklistů v hexu ---
gdf[COL_TOTAL_AVG] = (gdf[COL_TOTAL_TRP] * gdf[COL_TIME_S]) / float(ACTIVE_SECS)
gdf[COL_TOTAL_AVG] = gdf[COL_TOTAL_AVG].fillna(0.0)

# (volitelné) zaokrouhlení
# gdf[COL_TOTAL_TRP] = gdf[COL_TOTAL_TRP].round(3)
# gdf[COL_TOTAL_AVG] = gdf[COL_TOTAL_AVG].round(6)

# --- Uložení ---
gdf.to_file(OUTPUT_GPKG, layer=out_layer, driver="GPKG")
print(f"[OK] Uloženo do '{OUTPUT_GPKG}' (vrstva: '{out_layer}')")
print(f"     Přidané sloupce: {COL_TOTAL_TRP}, {COL_TOTAL_AVG}")
print(f"     Pravidlo: STRAVA >= UNIMOTION * (1 + {MERGE_THRESHOLD:.2f}) → sčítám")
# ============================================================================ 
