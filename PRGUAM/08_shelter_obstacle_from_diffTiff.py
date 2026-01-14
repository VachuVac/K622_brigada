# -*- coding: utf-8 -*-
"""
01_shelter_obstacle_from_diffTiff.py
--------------------------------
Varianta řízená seznamem percentilů.

Počítá se:
- DTM_hex_*        ... statistiky z band 2 (min, max, mean a definované percentily)
- Rel_height_hex_* ... statistiky z band 4 (min, max, mean a definované percentily)
- DSM_hex_*        ... statistiky z (band2 + band4) (min, max, mean a definované percentily)
- shelter_obstacle_rel = podíl pixelů, kde Rel_height >= H_MIN
- shelter_obstacle_mean  = binárně (Rel_height_hex_mean >= H_MIN)
- shelter_obstacle_pXX   = binárně (Rel_height_hex_pXX  >= H_MIN) pro všechny zvolená pXX

Skript je psaný tak, aby fungoval i na Windows (multiprocessing).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping

import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import matplotlib.pyplot as plt


# =========================
# KONFIGURACE / CESTY
# =========================
# VSTUPY
HEX_GPKG  = r"G:\Python_env_PragUAM\testing_location_01\h3_res12_testing_location_01_100m_5514.gpkg" 
HEX_LAYER = "h3_res12_polygons_testing_location_01"  # název vrstvy s hexy v GPKG

SHELTER_TIF = r"G:\Python_env_PragUAM\testing_location_01\testing_location_01_DSM_DTM_with_building_band.tif"
# band 2 = DTM, band 4 = Rel_height (height_diff_building); DSM = band2 + band4
# VÝSTUP
OUT_GPKG  = r"G:\Python_env_PragUAM\testing_location_01\testing_location_01_h3_shelter_obstacle.gpkg"


# # VSTUPY
# HEX_GPKG  = r"G:\Python_env_PragUAM\testing_location_03\h3_res12_polygons_testing_location_03_5514.gpkg" 
# HEX_LAYER = "h3_res12_polygons_testing_location_03"  # název vrstvy s hexy v GPKG

# SHELTER_TIF = r"G:\Python_env_PragUAM\testing_location_03\testing_location_03_DSM_DTM_with_building_band.tif"
# # band 2 = DTM, band 4 = Rel_height (height_diff_building); DSM = band2 + band4
# # VÝSTUP
# OUT_GPKG  = r"G:\Python_env_PragUAM\testing_location_03\testing_location_03_h3_shelter_obstacle.gpkg"


OUT_LAYER = "shelter_3D_combined"


# PRAH PRO OBSTACLE
H_MIN = 1.8  # m


PERCENTILES: List[float] = [0.25, 0.5, 0.75]
# Příklad: [0.5, 0.75, 0.9] nebo [0.25, 0.5, 0.75, 0.9]


# Paralelizace
TILE_SIZE = 2048
WORKERS   = "auto"  # None|"auto"|int


# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =========================
# POMOCNÍCI
# =========================

def pct_suffix(p: float) -> str:
    """0.75 -> 'p75'"""
    return f"p{int(round(p * 100))}"


# =========================
# P² KVANTIL (Jain & Chlamtac)
# =========================

class P2Quantile:
    """
    Online P² algoritmus pro odhad kvantilu bez ukládání všech dat.
    Implementace minimal: drží 5 markerů. Vhodné pro robustní a rychlé kvantily.
    """
    def __init__(self, prob: float):
        self.prob = float(prob)
        self.n = 0
        self.q = [0.0] * 5
        self.npos = [0] * 5
        self.dn = [0.0] * 5

    def add(self, x: float):
        if self.n < 5:
            self.q[self.n] = float(x)
            self.n += 1
            if self.n == 5:
                self.q.sort()
                self.npos = [1, 2, 3, 4, 5]
                p = self.prob
                self.dn = [0.0, p/2.0, p, (1.0+p)/2.0, 1.0]
            return

        # Najdi k, kam x padá
        if x < self.q[0]:
            self.q[0] = float(x)
            k = 0
        elif x >= self.q[4]:
            self.q[4] = float(x)
            k = 3
        else:
            k = 0
            for i in range(1, 5):
                if x < self.q[i]:
                    k = i - 1
                    break

        for i in range(k + 1, 5):
            self.npos[i] += 1
        for i in range(5):
            self.dn[i] += [0.0, 0.5*self.prob, self.prob, 0.5*(1+self.prob), 1.0][i]

        for i in [1, 2, 3]:
            d = self.dn[i] - self.npos[i]
            if (d >= 1 and self.npos[i+1] - self.npos[i] > 1) or (d <= -1 and self.npos[i-1] - self.npos[i] < -1):
                d = int(np.sign(d))
                qi = self.q[i] + d * (
                    (self.q[i+1] - self.q[i]) / (self.npos[i+1] - self.npos[i])
                    + (self.q[i-1] - self.q[i]) / (self.npos[i-1] - self.npos[i])
                )
                if self.q[i-1] < qi < self.q[i+1]:
                    self.q[i] = qi
                else:
                    self.q[i] += d * (self.q[i + d] - self.q[i]) / (self.npos[i + d] - self.npos[i])
                self.npos[i] += d

    def result(self) -> float:
        if self.n == 0:
            return float("nan")
        if self.n < 5:
            arr = sorted(self.q[:self.n])
            idx = max(0, min(len(arr) - 1, int(round(self.prob * (len(arr) - 1)))))
            return float(arr[idx])
        # jednoduché čtení markeru poblíž kvantilu – pro 0.5 by to byl q[2]
        return float(self.q[int(round(self.prob*4))])


# =========================
# AGREGÁTOR ZÓN — DYNAMICKÉ PERCENTILY
# =========================

class ZoneStats:
    def __init__(self, want_rel: bool = False):
        self.min = float("inf")
        self.max = -float("inf")
        self.count = 0
        self.sum = 0.0
        self.want_rel = want_rel
        self.ge_count = 0  # počet hodnot >= rel_threshold
        self.quants: Dict[float, P2Quantile] = {p: P2Quantile(p) for p in PERCENTILES}

    def update(self, arr, rel_threshold=None):
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            return
        self.count += a.size
        self.sum += float(a.sum())
        amin = float(a.min()); amax = float(a.max())
        if amin < self.min: self.min = amin
        if amax > self.max: self.max = amax
        # P2 kvantily – feed po prvcích
        for v in a.ravel():
            for q in self.quants.values():
                q.add(v)
        if self.want_rel and (rel_threshold is not None):
            self.ge_count += int((a >= rel_threshold).sum())


# =========================
# RASTERIZACE ZÓN
# =========================

def build_zone_raster(hex_gpkg: str, hex_layer: str, raster_like_path: str) -> Tuple[np.ndarray, np.ndarray, gpd.GeoDataFrame]:
    gdf = gpd.read_file(hex_gpkg, layer=hex_layer)
    logger.info(f"Načteno {len(gdf)} hexů. CRS: {gdf.crs}")

    with rasterio.open(raster_like_path) as src:
        r_crs = src.crs
        r_transform = src.transform
        out_shape = (src.height, src.width)

    if gdf.crs is None or r_crs is None:
        raise ValueError("Chybí CRS u vektorových nebo rastrových dat.")

    if gdf.crs != r_crs:
        gdf = gdf.to_crs(r_crs)

    if "zone_id" not in gdf.columns:
        gdf = gdf.reset_index(drop=True)
        gdf["zone_id"] = (np.arange(len(gdf)) + 1).astype(np.int32)

    shapes = [(mapping(geom), int(zid)) for geom, zid in zip(gdf.geometry, gdf["zone_id"]) if geom is not None and not geom.is_empty]

    zones_arr = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=r_transform,
        fill=0,
        all_touched=False,
        dtype="int32",
    )

    return zones_arr, gdf["zone_id"].to_numpy(dtype=np.int32), gdf


# =========================
# PRŮCHOD OKNEM – 1 BAND
# =========================

def _process_window(raster_path, band, window: Window, nodata_val, zones_tile: np.ndarray, want_rel: bool, rel_threshold):
    zones_sub = zones_tile
    stats: Dict[int, ZoneStats] = {}

    with rasterio.open(raster_path) as src:
        data = src.read(band, window=window)
        msk  = src.read_masks(band, window=window)  # 0 = NoData, >0 = valid

    if data.shape != zones_sub.shape or msk.shape != zones_sub.shape:
        raise ValueError(f"Zónové okno a data/maska nemají stejný tvar: data={data.shape}, mask={msk.shape}, zones={zones_sub.shape}")

    valid_mask = (msk > 0) & (zones_sub > 0)
    if not np.any(valid_mask):
        return {}

    valid_zones = zones_sub[valid_mask]
    valid_data  = data[valid_mask]

    for u in np.unique(valid_zones):
        vals = valid_data[valid_zones == u]
        if vals.size == 0:
            continue
        st = stats.get(int(u))
        if st is None:
            st = ZoneStats(want_rel=want_rel)
            stats[int(u)] = st
        st.update(vals, rel_threshold=rel_threshold)

    # vracíme primitiva + list kvantilů v pořadí PERCENTILES
    out: Dict[int, Tuple[float, float, int, float, int | None, List[float]]] = {}
    for zid, st in stats.items():
        qvals = [st.quants[p].result() for p in PERCENTILES]
        out[zid] = (st.min, st.max, st.count, st.sum, st.ge_count if want_rel else None, qvals)
    return out


def reduce_stats(global_stats: Dict[int, ZoneStats], partial: Dict[int, Tuple[Any, ...]], want_rel: bool):
    def _init_markers(vmin, pxx, vmax):
        m1 = float(vmin); m3 = float(pxx); m5 = float(vmax)
        m2 = (m1 + m3) / 2.0; m4 = (m3 + m5) / 2.0
        return [m1, m2, m3, m4, m5], [1, 2, 3, 4, 5]

    def _merge_markers(curr_q, vmin, pxx, vmax):
        m1 = min(curr_q[0], float(vmin))
        m5 = max(curr_q[4], float(vmax))
        m3 = (curr_q[2] + float(pxx)) / 2.0
        m2 = (m1 + m3) / 2.0; m4 = (m3 + m5) / 2.0
        return [m1, m2, m3, m4, m5], [1, 2, 3, 4, 5]

    for zid, tup in partial.items():
        vmin, vmax, cnt, ssum, ge, qvals = tup
        zs = global_stats.get(zid)
        if zs is None:
            zs = ZoneStats(want_rel=want_rel)
            zs.min = float(vmin); zs.max = float(vmax)
            zs.count = int(cnt);   zs.sum = float(ssum)
            if want_rel:
                zs.ge_count = 0 if ge is None else int(ge)
            # init všech percentilů
            for p, qv in zip(PERCENTILES, qvals):
                zs.quants[p].q, zs.quants[p].npos = _init_markers(vmin, qv, vmax)
                zs.quants[p].n = 5
            global_stats[zid] = zs
        else:
            if float(vmin) < zs.min: zs.min = float(vmin)
            if float(vmax) > zs.max: zs.max = float(vmax)
            zs.count += int(cnt); zs.sum += float(ssum)
            if want_rel and ge is not None:
                zs.ge_count += int(ge)
            for p, qv in zip(PERCENTILES, qvals):
                zs.quants[p].q, zs.quants[p].npos = _merge_markers(zs.quants[p].q, vmin, qv, vmax)
                zs.quants[p].n = 5


def process_raster_per_zone_parallel(
    raster_path: str,
    band: int,
    zones_arr: np.ndarray,
    want_rel: bool,
    rel_threshold: float | None,
    nodata_val=None,
    tile: int = TILE_SIZE,
    workers: int | str | None = WORKERS,
):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width

    if workers is None or workers == "auto":
        try:
            workers = max(1, (os.cpu_count() or 4) - 1)
        except Exception:
            workers = 1

    windows = []
    for r0 in range(0, height, tile):
        for c0 in range(0, width, tile):
            h = min(tile, height - r0)
            w = min(tile, width - c0)
            win = Window(c0, r0, w, h)
            ztile = zones_arr[r0:r0 + h, c0:c0 + w]
            windows.append((win, ztile))

    agg: Dict[int, ZoneStats] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(_process_window, raster_path, band, win, nodata_val, ztile, want_rel, rel_threshold)
            for (win, ztile) in windows
        ]
        for fut in as_completed(futures):
            part = fut.result()
            reduce_stats(agg, part, want_rel=want_rel)
    return agg


# =========================
# PRŮCHOD OKNEM – 2 BANDY (součet: DSM = band2 + band4)
# =========================

def _process_window_two_bands_sum(raster_path, band_a, band_b, window: Window, nodata_val_a, nodata_val_b, zones_tile: np.ndarray):
    zones_sub = zones_tile
    stats: Dict[int, ZoneStats] = {}

    with rasterio.open(raster_path) as src:
        a   = src.read(band_a, window=window)
        b   = src.read(band_b, window=window)
        ma  = src.read_masks(band_a, window=window)
        mb  = src.read_masks(band_b, window=window)

    if a.shape != zones_sub.shape or b.shape != zones_sub.shape or ma.shape != zones_sub.shape or mb.shape != zones_sub.shape:
        raise ValueError(f"Nesoulad tvarů pro DSM: a={a.shape}, b={b.shape}, ma={ma.shape}, mb={mb.shape}, zones={zones_sub.shape}")

    mask = (ma > 0) & (mb > 0)
    data = a.astype("float32") + b.astype("float32")
    valid_mask = mask & (zones_sub > 0)
    if not np.any(valid_mask):
        return {}

    valid_zones = zones_sub[valid_mask]
    valid_data  = data[valid_mask]

    for u in np.unique(valid_zones):
        vals = valid_data[valid_zones == u]
        if vals.size == 0:
            continue
        st = stats.get(int(u))
        if st is None:
            st = ZoneStats(want_rel=False)
            stats[int(u)] = st
        st.update(vals, rel_threshold=None)

    out: Dict[int, Tuple[float, float, int, float, None, List[float]]] = {}
    for zid, st in stats.items():
        qvals = [st.quants[p].result() for p in PERCENTILES]
        out[zid] = (st.min, st.max, st.count, st.sum, None, qvals)
    return out


def process_two_bands_per_zone_parallel(
    raster_path: str,
    band_a: int,
    band_b: int,
    zones_arr: np.ndarray,
    nodata_val_a=None,
    nodata_val_b=None,
    tile: int = TILE_SIZE,
    workers: int | str | None = WORKERS,
):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width

    if workers is None or workers == "auto":
        try:
            workers = max(1, (os.cpu_count() or 4) - 1)
        except Exception:
            workers = 1

    windows = []
    for r0 in range(0, height, tile):
        for c0 in range(0, width, tile):
            h = min(tile, height - r0)
            w = min(tile, width - c0)
            win = Window(c0, r0, w, h)
            ztile = zones_arr[r0:r0 + h, c0:c0 + w]
            windows.append((win, ztile))

    agg: Dict[int, ZoneStats] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(_process_window_two_bands_sum, raster_path, band_a, band_b, win, nodata_val_a, nodata_val_b, ztile)
            for (win, ztile) in windows
        ]
        for fut in as_completed(futures):
            part = fut.result()
            reduce_stats(agg, part, want_rel=False)
    return agg


# =========================
# PŘEVOD STATISTIK NA DATAFRAME
# =========================

def _stats_to_df(stats_dict: Dict[int, ZoneStats] | Dict[int, Tuple[Any, ...]], prefix: str, include_rel: bool = False) -> pd.DataFrame:
    rows = []
    for zid, st in stats_dict.items():
        if isinstance(st, ZoneStats):
            vmin, vmax, cnt, ssum, ge = st.min, st.max, st.count, st.sum, st.ge_count
            qvals = [st.quants[p].result() for p in PERCENTILES]
        else:
            vmin, vmax, cnt, ssum, ge, qvals = st

        mean = (ssum / cnt) if cnt else np.nan
        rec = {
            "zone_id": zid,
            f"{prefix}_min": vmin,
            f"{prefix}_max": vmax,
            f"{prefix}_mean": mean,
        }
        for p, qv in zip(PERCENTILES, qvals):
            rec[f"{prefix}_{pct_suffix(p)}"] = qv
        if include_rel:
            rec[f"{prefix}_ge_count"] = 0 if ge is None else ge
            rec[f"{prefix}_count"] = cnt
        rows.append(rec)
    df = pd.DataFrame(rows).set_index("zone_id")
    return df


# =========================
# DIAGNOSTIKA – histogram jednoho hexu
# =========================

def gather_band_values_for_zone(raster_path, band, zones_arr, target_zone_id, tile=TILE_SIZE):
    """Sesbírá všechny hodnoty daného bandu pro konkrétní zone_id (bez NoData, nuly se ponechají)."""
    vals = []
    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width
        for r0 in range(0, height, tile):
            for c0 in range(0, width, tile):
                h = min(tile, height - r0)
                w = min(tile, width - c0)
                win = Window(c0, r0, w, h)
                ztile = zones_arr[r0:r0 + h, c0:c0 + w]
                if not (ztile == target_zone_id).any():
                    continue
                data = src.read(band, window=win)
                msk  = src.read_masks(band, window=win)  # 0 = NoData
                m = (msk > 0) & (ztile == target_zone_id)
                if m.any():
                    vals.append(data[m].astype("float32"))
    if vals:
        return np.concatenate(vals)
    return np.array([], dtype="float32")


# =========================
# MAIN
# =========================

def main():
    # --- Načtení rasteru a zón ---
    with rasterio.open(SHELTER_TIF) as src:
        sh_nodata = src.nodata
    if sh_nodata is None:
        sh_nodata = -9999

    zones_arr, zone_ids, hex_gdf = build_zone_raster(HEX_GPKG, HEX_LAYER, SHELTER_TIF)

    logger.info("Rasterizace zón → paměťové pole (bez ukládání na disk)")

    # --- Výpočty (jen SHELTER_TIF) ---
    logger.info(f"Paralelně zpracovávám: {Path(SHELTER_TIF).name} (band=2 DTM), tile={TILE_SIZE}, workers={WORKERS}")
    dtm_stats = process_raster_per_zone_parallel(
        SHELTER_TIF, 2, zones_arr, want_rel=False, rel_threshold=None,
        nodata_val=sh_nodata, tile=TILE_SIZE, workers=WORKERS
    )
    logger.info("DTM hotovo; zón: %d", len(dtm_stats))

    logger.info("Paralelně zpracovávám: Rel_height (band=4)")
    rel_stats = process_raster_per_zone_parallel(
        SHELTER_TIF, 4, zones_arr, want_rel=True, rel_threshold=H_MIN,
        nodata_val=sh_nodata, tile=TILE_SIZE, workers=WORKERS
    )
    logger.info("Rel hotovo; zón: %d", len(rel_stats))

    logger.info("Paralelně zpracovávám: DSM = band2 + band4")
    dsm_stats = process_two_bands_per_zone_parallel(
        SHELTER_TIF, 2, 4, zones_arr, nodata_val_a=sh_nodata, nodata_val_b=sh_nodata,
        tile=TILE_SIZE, workers=WORKERS
    )
    logger.info("DSM hotovo; zón: %d", len(dsm_stats))

    # --- Převod na tabulku ---
    dtm_df = _stats_to_df(dtm_stats, "DTM_hex", include_rel=False)
    rel_df = _stats_to_df(rel_stats, "Rel_height_hex", include_rel=True)
    dsm_df = _stats_to_df(dsm_stats, "DSM_hex", include_rel=False)

    # --- Složení tabulky výsledků ---
    out_df = dtm_df.join(dsm_df, how="outer").join(rel_df, how="outer")

    # 1) podíl pixelů >= H_MIN (0–1)
    out_df["shelter_obstacle_rel"] = (
        rel_df["Rel_height_hex_ge_count"] / rel_df["Rel_height_hex_count"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 2) binární rozhodnutí podle agregovaných hodnot hexu (1 pokud >= H_MIN)
    out_df["shelter_obstacle_mean"] = (rel_df["Rel_height_hex_mean"] >= H_MIN).astype(int)
    for p in PERCENTILES:
        out_df[f"shelter_obstacle_{pct_suffix(p)}"] = (rel_df[f"Rel_height_hex_{pct_suffix(p)}"] >= H_MIN).astype(int)

    # --- Připojení zpět k hexům ---
    if "zone_id" not in hex_gdf.columns:
        raise RuntimeError("hex_gdf neobsahuje sloupec 'zone_id'.")
    hex_gdf = hex_gdf.set_index("zone_id").join(out_df, how="left").reset_index()

    # --- Diagnostika: histogram Rel_height pro 10 náhodných hexů ---
    valid_zone_ids = rel_df[rel_df["Rel_height_hex_count"] > 0].index.values
    if valid_zone_ids.size:
        # vybereme až 10 náhodných unikátních ID (ne víc než počet zón)
        n_samples = min(10, valid_zone_ids.size)
        sampled_ids = np.random.choice(valid_zone_ids, size=n_samples, replace=False)

        for z_id in sampled_ids:
            z_id = int(z_id)
            rel_vals = gather_band_values_for_zone(SHELTER_TIF, 4, zones_arr, z_id, tile=TILE_SIZE)
            if rel_vals.size == 0:
                continue

            stats_row = rel_df.loc[z_id]
            mean_v = float(stats_row["Rel_height_hex_mean"])

            fig = plt.figure(figsize=(8, 4.5))
            counts, bins, _ = plt.hist(rel_vals, bins=50)
            ymax = float(counts.max() if counts.size else 1.0)
            plt.ylim(0, ymax * 1.18)

            # mean
            plt.axvline(mean_v, linestyle="-", linewidth=2, color="C0", label=f"mean = {mean_v:.2f} m")
            plt.text(mean_v, ymax * 1.02, f"{mean_v:.2f}", rotation=90, va="bottom", ha="center", color="C0")

            # všechny percentily
            for i, p in enumerate(PERCENTILES, start=1):
                val = float(stats_row[f"Rel_height_hex_{pct_suffix(p)}"])
                color = f"C{(i % 9)}"
                plt.axvline(val, linestyle="--", linewidth=2, color=color, label=f"{pct_suffix(p)} = {val:.2f} m")
                plt.text(val, ymax * 1.02, f"{val:.2f}", rotation=90, va="bottom", ha="center", color=color)

            # práh H_MIN
            plt.axvline(H_MIN, linestyle=":", linewidth=2, color="C8", label=f"H_MIN = {H_MIN:.2f} m")
            plt.text(H_MIN, ymax * 1.02, f"{H_MIN:.2f}", rotation=90, va="bottom", ha="center", color="C8")

            plt.title(f"Rel. height distribution – zone_id {z_id}")
            plt.xlabel("Relative height [m]")
            plt.ylabel("Pixel count")
            plt.legend(loc="upper center")

            out_png = Path(OUT_GPKG).with_name(f"diagnostics_rel_hist_zone_{z_id}.png")
            fig.savefig(out_png, dpi=160, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Uložen histogram: {out_png}")
    else:
        logger.info("Diagnostika přeskočena – žádný hex s Rel_height daty.")


    # --- Zápis do GPKG (bez if_exists) ---
    out_path = Path(OUT_GPKG)
    if out_path.exists():
        out_path.unlink()
    hex_gdf.to_file(OUT_GPKG, layer=OUT_LAYER, driver="GPKG")
    logger.info("Zapsáno: %s | vrstva: %s", OUT_GPKG, OUT_LAYER)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        sys.exit(1)
