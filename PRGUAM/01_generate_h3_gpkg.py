# -*- coding: utf-8 -*-
"""
Generate H3 hex grid from polygons in a GPKG layer and save as Shapefile (EPSG:4326).

Requirements:
    pip install h3>=4 geopandas pyogrio shapely pyproj

Notes:
    - Input layer (IN_GPKG/IN_LAYER) can be in WGS84/ETRS/5514/...; it will be reprojected to EPSG:4326.
    - Output is ALWAYS in EPSG:4326.
    - Shapefile field-name limit: keep attributes short; here we write only 'h3' and 'res'.
"""

import os
from pathlib import Path
import geopandas as gpd
import pyogrio
from shapely.geometry import Polygon
from pyproj import CRS
import h3

# =======================
# CONFIG — EDIT THESE
# =======================
IN_GPKG  = r"G:\Python_env_PragUAM\DSM_DTM\polygons_testing_location_01_100m.gpkg"
IN_LAYER = "polygons_testing_location_01_100m"
BASE_EXPORT = r"G:\Python_env_PragUAM\testing_location_01\h3_res12_testing_location_01_100m"
RES = 12  # e.g. 8–12 for city-scale


# IN_GPKG  = r"G:\Python_env_PragUAM\DSM_DTM\polygons_praha_100m.gpkg"
# IN_LAYER = "polygons_praha_100m"

# OUT_SHP  = r"G:\Python_env_PragUAM\DSM_DTM\h3_hexes\h3_res12_polygons_praha_100m.shp"
# RES = 12  # e.g. 8–12 for city-scale

# IN_GPKG  = r"G:\Python_env_PragUAM\testing_location_03\testing_location_03_100m.gpkg"
# IN_LAYER = "polygons"

# OUT_SHP  = r"G:\Python_env_PragUAM\testing_location_03\h3_res12_polygons_testing_location_03.shp"
# RES = 12  # e.g. 8–12 for city-scale

base = Path(BASE_EXPORT)

OUT_SHP_4326   = BASE_EXPORT + ".shp"
OUT_GPKG_5514  = BASE_EXPORT + "_5514.gpkg"
OUT_GPKG_LAYER = Path(BASE_EXPORT).name
# GPKG jde do nadřazené složky "testing_location_01"


OUT_GPKG_LAYER = base.stem.replace("testing_location_01_100m", "polygons_testing_location_01")

# =======================
# HELPERS
# =======================
def _to_4326(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise RuntimeError("Input layer has no CRS defined. Please define CRS in the source data.")
    if CRS.from_user_input(gdf.crs).to_epsg() == 4326:
        return gdf
    return gdf.to_crs(epsg=4326)

def _cell_to_polygon(h: str) -> Polygon:
    # h3.cell_to_boundary returns [(lat, lng), ...]; convert to (lon, lat)
    boundary = h3.cell_to_boundary(h)
    coords = [(lng, lat) for (lat, lng) in boundary]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return Polygon(coords)

def _remove_existing(path: str):
    p = Path(path)

    if not p.exists():
        return

    try:
        if p.suffix.lower() == ".shp":
            base = p.with_suffix("")
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                f = base.with_suffix(ext)
                if f.exists():
                    f.unlink()
                    print(f"[clean] removed {f}")

        elif p.suffix.lower() == ".gpkg":
            p.unlink()
            print(f"[clean] removed {p}")

        else:
            raise ValueError(f"Unsupported format: {p.suffix}")

    except PermissionError as e:
        raise RuntimeError(
            f"Cannot remove '{p}'. "
            f"Is the file opened in QGIS or another application?"
        ) from e

# =======================
# MAIN
# =======================
def main():
    in_path = Path(IN_GPKG)
    if not in_path.exists():
        raise FileNotFoundError(f"Input GPKG not found: {IN_GPKG}")

    print(f"[info] Input:  {IN_GPKG} | layer '{IN_LAYER}'")
    print(f"[info] Output: {OUT_SHP_4326} (Shapefile, EPSG:4326)")
    print(f"[info] Output: {OUT_GPKG_5514} (GPKG, EPSG:5514)")
    print(f"[info] H3 resolution: {RES}")

    # Read polygons from GPKG
    gdf_in = gpd.read_file(IN_GPKG, layer=IN_LAYER)
    if gdf_in.empty:
        raise RuntimeError("Input layer contains no features.")

    print(f"[crs] Input layer CRS: {gdf_in.crs}")
    # Reproject to EPSG:4326 for H3
    gdf_ll = _to_4326(gdf_in)

    # Prepare GeoJSON-like geometries for H3 (lon/lat order is expected by h3 v4)
    geoms_ll = [geom.__geo_interface__ for geom in gdf_ll.geometry if geom is not None and not geom.is_empty]

    # Generate H3 cells covering all polygons
    cells_all = set()
    for gj_geom in geoms_ll:
        shape = h3.geo_to_h3shape(gj_geom)  # expects lon/lat geojson geometry
        cells = h3.h3shape_to_cells(shape, RES)
        cells_all.update(cells)

    print(f"[info] H3 cells total (res={RES}): {len(cells_all):,}")

    # Build hex polygons in EPSG:4326
    hex_polys = [_cell_to_polygon(h) for h in cells_all]
    gdf_hex = gpd.GeoDataFrame(
        {"h3": list(cells_all), "res": RES},
        geometry=hex_polys,
        crs="EPSG:4326"
    )

    # Ensure output folder exists
    Path(OUT_SHP_4326).parent.mkdir(parents=True, exist_ok=True)

    
    # WRITE OUTPUTS
    # =======================

    # --- SHP v EPSG:4326 ---
    Path(OUT_SHP_4326).parent.mkdir(parents=True, exist_ok=True)
    _remove_existing(OUT_SHP_4326)

    gdf_hex_4326 = gdf_hex  # už je v EPSG:4326
    gdf_hex_4326.to_file(
        OUT_SHP_4326,
        driver="ESRI Shapefile",
        encoding="utf-8"
    )

    print(f"[done] SHP 4326: {OUT_SHP_4326}")


    # --- GPKG v EPSG:5514 ---
    Path(OUT_GPKG_5514).parent.mkdir(parents=True, exist_ok=True)
    _remove_existing(OUT_GPKG_5514)

    gdf_hex_5514 = gdf_hex_4326.to_crs(epsg=5514)
    gdf_hex_5514.to_file(
        OUT_GPKG_5514,
        layer=OUT_GPKG_LAYER,
        driver="GPKG"
    )

    print(f"[done] GPKG 5514: {OUT_GPKG_5514} | layer='{OUT_GPKG_LAYER}'")


if __name__ == "__main__":
    main()
