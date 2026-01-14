# dmr5g_atom_download.py
import os
import math
import time
import zipfile
import io
import requests
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from xml.etree import ElementTree as ET

# ---- Nastavení ----
# BOUNDARY_PATH = r"G:\Python_env_PragUAM\testing_location_03\h3_res12_polygons_testing_location_03_5514.gpkg"        # nebo .gpkg; vrstva s polygonem HMP Praha
BOUNDARY_PATH = r"G:\Python_env_PragUAM\DSM_DTM\polygons_testing_location_01_100m.gpkg"        # nebo .gpkg; vrstva s polygonem HMP Praha

#5G! DMR
OUT_DIR = "testing_location_01\data_dmr5g_testing_location_01"       # cílová složka pro ZIP (LAZ) dlaždice
# OUT_DIR = "testing_location_03\data_dmr5g_testing_location_03" 
THEME = "DMR5G-SJTSK"              # série DMR 5G v S-JTSK


#1G! DSM
OUT_DIR = "testing_location_01\data_dmr1g_testing_location_01"
# OUT_DIR = "testing_location_03\data_dmr1g_testing_location_03"
THEME = "DMP1G-SJTSK"              # série DMP 1G v S-JTSK


BASE_GET = "https://atom.cuzk.cz/get.ashx"
# timeouty a opakování
HTTP_TIMEOUT = 60
RETRY = 3
SLEEP_BETWEEN = 0.8  # šetří server

# ---- Pomocné funkce ----
def fetch_atom_entries(params):
    """Vrátí seznam <entry> elementů z ATOM feedu pro dané parametry (dict)."""
    for attempt in range(RETRY):
        try:
            r = requests.get(BASE_GET, params=params, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            # ČÚZK vrací application/atom+xml; zpracujeme XML
            return ET.fromstring(r.content)
        except Exception as e:
            if attempt == RETRY - 1:
                raise
            time.sleep(1.5)
    return None

def parse_entries(atom_root):
    ns = {"a": "http://www.w3.org/2005/Atom", "georss":"http://www.georss.org/georss"}
    # v <a:title> bývá "Response entries count = N" – zkusíme získat počet
    count = None
    title_el = atom_root.find("a:title", ns)
    if title_el is not None and "count" in (title_el.text or "").lower():
        try:
            count = int(title_el.text.split("=")[-1].strip())
        except:
            pass
    entries = []
    for e in atom_root.findall("a:entry", ns):
        link = e.find("a:link", ns)
        href = link.get("href") if link is not None else None
        # pro přesnější filtraci si přečteme i georss polygon dlaždice
        geopol = e.find("georss:polygon", ns)
        poly = None
        if geopol is not None and geopol.text:
            coords = [float(x) for x in geopol.text.split()]
            # georss:polygon = lat lon lat lon ...
            pts = [(coords[i+1], coords[i]) for i in range(0, len(coords), 2)]
            from shapely.geometry import Polygon
            poly = Polygon(pts)
        entries.append({"href": href, "poly_wgs84": poly})
    return entries, count

def download_file(url, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    name = url.split("/")[-1]
    out_path = os.path.join(out_dir, name)
    if os.path.exists(out_path):
        return out_path
    for attempt in range(RETRY):
        try:
            with requests.get(url, stream=True, timeout=HTTP_TIMEOUT) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1<<20):
                        if chunk:
                            f.write(chunk)
            return out_path
        except Exception:
            if attempt == RETRY - 1:
                raise
            time.sleep(1.5)
    return out_path

def query_bbox(bbox_wgs84, theme=THEME, extra_params=None):
    params = {"theme": theme,
              "bbox": f"{bbox_wgs84[0]},{bbox_wgs84[1]},{bbox_wgs84[2]},{bbox_wgs84[3]}"}
    if extra_params:
        params.update(extra_params)
    atom = fetch_atom_entries(params)
    entries, count = parse_entries(atom)
    return entries, (count if count is not None else len(entries))

def subdivide_bbox(bbox, nx=2, ny=2):
    minx, miny, maxx, maxy = bbox
    dx = (maxx - minx) / nx
    dy = (maxy - miny) / ny
    boxes = []
    for i in range(nx):
        for j in range(ny):
            boxes.append((minx + i*dx, miny + j*dy, minx + (i+1)*dx, miny + (j+1)*dy))
    return boxes

# ---- Hlavní logika ----
def main():
    # 1) Načti hranici Prahy a převeď do WGS84
    gdf = gpd.read_file(BOUNDARY_PATH)
    if gdf.empty:
        raise RuntimeError("Hranice nebyla načtena.")
    boundary = unary_union(gdf.to_crs(4326).geometry)  # WGS-84 polygon(y)
    bbox = boundary.bounds  # (minx, miny, maxx, maxy) ve WGS-84

    # 2) Rekurzivně děl BBOX, dokud počet výsledků < 100 (limit služby)
    stack = [bbox]
    hrefs = set()
    while stack:
        bb = stack.pop()
        entries, count = query_bbox(bb)
        if count >= 100:
            # rozdělíme na 4 menší boxy a znovu
            stack.extend(subdivide_bbox(bb, 2, 2))
            continue
        # filtr: stáhni jen dlaždice, které opravdu kříží polygon Prahy
        for e in entries:
            if not e["href"]:
                continue
            poly = e["poly_wgs84"]
            if poly is None or poly.intersects(boundary):
                hrefs.add(e["href"])
        time.sleep(SLEEP_BETWEEN)

    print(f"Nalezeno dlaždic: {len(hrefs)}")
    # 3) Stahuj
    for url in sorted(hrefs):
        print("Stahuji:", url)
        path = download_file(url, OUT_DIR)
        print("OK ->", path)
        time.sleep(SLEEP_BETWEEN)

    print("Hotovo.")

if __name__ == "__main__":
    main()
