#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LAS/LAZ složka (EPSG:5514) -> GeoTIFF 1 m/px (max Z) přes PDAL/GDAL CLI, paralelně.
- NOVĚ: umí vstupní *složku* – načte všechny LAS/LAZ, sjednotí je a dlaždicuje.
- BEZ Python bindingu pdal; stačí pdal.exe a gdal* v PATH.
- SRS řeší readers.las (override_srs). writers.gdal už 'srs' NEBERE.
- Verbose logování, možnost ponechat JSON pipeline.
- AUTO-RESUME: přeskakuje už spočítané dlaždice (lze vypnout --no-resume).
"""

import argparse, json, math, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import laspy
from typing import Iterable, List, Tuple

Bounds = Tuple[float, float, float, float]


def die(msg):
    print(f"[ERR] {msg}", file=sys.stderr)
    sys.exit(1)

def have(binname: str) -> bool:
    return shutil.which(binname) is not None


def list_las_inputs(path: str) -> List[str]:
    """Vrátí seznam vstupních LAS/LAZ. path může být soubor nebo složka."""
    if os.path.isdir(path):
        files = [
            os.path.join(path, p)
            for p in os.listdir(path)
            if p.lower().endswith((".las", ".laz"))
        ]
        files.sort()
        return files
    if os.path.isfile(path) and path.lower().endswith((".las", ".laz")):
        return [os.path.abspath(path)]
    return []


def get_bounds_las(path: str) -> Bounds:
    with laspy.open(path) as f:
        h = f.header
        return float(h.mins[0]), float(h.mins[1]), float(h.maxs[0]), float(h.maxs[1])


def get_bounds_many(paths: Iterable[str]) -> Bounds:
    it = iter(paths)
    try:
        first = next(it)
    except StopIteration:
        die("Nenalezeny žádné LAS/LAZ vstupy.")
    minx, miny, maxx, maxy = get_bounds_las(first)
    for p in it:
        a, b, c, d = get_bounds_las(p)
        minx = min(minx, a)
        miny = min(miny, b)
        maxx = max(maxx, c)
        maxy = max(maxy, d)
    return (minx, miny, maxx, maxy)


def run(cmd, echo=True):
    if echo:
        print("[CMD]", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.stdout:
        print("[OUT]", p.stdout.strip())
    if p.returncode != 0:
        err = (p.stderr or "").strip()
        if err:
            print("[ERR]", err, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)} (code {p.returncode})\nSTDERR:\n{err}")
    if p.stderr:
        print("[WARN]", p.stderr.strip(), file=sys.stderr)
    return p.stdout


def build_pipeline_json(infiles: List[str], outfile: str, bounds: Bounds, res: float, nodata: float,
                        srs: str, classification: int | None = None) -> str:
    """Vytvoří PDAL pipeline. Podporuje 1 i N vstupních LAS/LAZ. Více vstupů se spojí přes filters.merge."""
    (minx, miny, maxx, maxy) = bounds
    bstr = f"([{minx},{maxx}],[{miny},{maxy}])"

    stages: list[dict] = []
    # 1..N readers
    for f in infiles:
        stages.append({"type": "readers.las", "filename": f, "override_srs": srs})
    # merge jen když je vstupů více
    if len(infiles) > 1:
        stages.append({"type": "filters.merge"})

    if classification is not None:
        stages.append({"type": "filters.range", "limits": f"Classification[{classification}:{classification}]"})

    stages.append({"type": "filters.crop", "bounds": bstr})

    stages.append({
        "type": "writers.gdal",
        "filename": outfile,
        "gdaldriver": "GTiff",
        "data_type": "float32",
        "resolution": res,
        "dimension": "Z",
        "output_type": "max",
        "nodata": nodata,
        "gdalopts": "TILED=YES,COMPRESS=DEFLATE,PREDICTOR=2,BIGTIFF=YES,NUM_THREADS=ALL_CPUS,BLOCKXSIZE=512,BLOCKYSIZE=512",
    })
    return json.dumps({"pipeline": stages}, ensure_ascii=False)


def tile_outpath(outdir, idx, bounds):
    minx, miny, _, _ = bounds
    return os.path.join(outdir, f"tile_{idx}_{int(minx)}_{int(miny)}.tif")


def is_tile_done(path):
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def make_empty_tile(outfile, bounds, res, srs, nodata):
    import math, os, tempfile, subprocess
    minx, miny, maxx, maxy = bounds
    cols = int(math.ceil((maxx - minx) / res))
    rows = int(math.ceil((maxy - miny) / res))
    if cols <= 0 or rows <= 0:
        raise RuntimeError(f"Empty tile size computed: {cols}x{rows} for bounds {bounds}")

    with tempfile.TemporaryDirectory() as td:
        asc = os.path.join(td, "empty.asc")
        header = (
            f"ncols         {cols}\n"
            f"nrows         {rows}\n"
            f"xllcorner     {minx}\n"
            f"yllcorner     {miny}\n"
            f"cellsize      {res}\n"
            f"NODATA_value  {nodata}\n"
        )
        line = ("{0} ".format(nodata)) * (cols - 1) + f"{nodata}\n"
        with open(asc, "w", encoding="utf-8") as f:
            f.write(header)
            for _ in range(rows):
                f.write(line)

        subprocess.check_call([
            "gdal_translate", asc, outfile,
            "-a_srs", srs,
            "-a_nodata", str(nodata),
            "-ot", "Float32",
            "-of", "GTiff",
            "-co", "TILED=YES",
            "-co", "COMPRESS=DEFLATE",
            "-co", "PREDICTOR=2",
            "-co", "BIGTIFF=YES",
            "-co", "BLOCKXSIZE=512",
            "-co", "BLOCKYSIZE=512",
        ])


def ensure_float32(path, nodata):
    try:
        out = subprocess.check_output(["gdalinfo", "-json", path], text=True)
        info = json.loads(out)
        btype = info["bands"][0]["type"]
    except Exception:
        btype = None
    if btype == "Float32":
        return
    tmp = path + ".f32.tif"
    subprocess.check_call([
        "gdal_translate", path, tmp,
        "-ot", "Float32",
        "-a_nodata", str(nodata),
        "-co", "TILED=YES",
        "-co", "COMPRESS=DEFLATE",
        "-co", "PREDICTOR=2",
        "-co", "BIGTIFF=YES",
        "-co", "BLOCKXSIZE=512",
        "-co", "BLOCKYSIZE=512",
    ])
    os.replace(tmp, path)


def worker(args):
    infiles, tile_bounds, res, nodata, srs, outdir, idx, classification, pdal_verbose, keep_json, resume = args
    outfile = tile_outpath(outdir, idx, tile_bounds)

    if resume and is_tile_done(outfile):
        print(f"[SKIP] Už hotovo → {outfile}")
        return outfile

    pipe_json = build_pipeline_json(infiles, outfile, tile_bounds, res, nodata, srs, classification)
    base_for_json = os.path.dirname(outdir)
    tmpjson_dir = os.path.join(base_for_json, "tmp_json")
    os.makedirs(tmpjson_dir, exist_ok=True)

    jpath = os.path.join(tmpjson_dir, f"tile_{idx}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        f.write(pipe_json)

    try:
        run(["pdal", "pipeline", "-v", str(pdal_verbose), "--debug", jpath], echo=True)
    except RuntimeError as e:
        msg = str(e)
        if "Unable to write GDAL data with no points for output" in msg:
            print(f"[INFO] Tile bez bodů → tvořím NoData raster: {outfile}")
            make_empty_tile(outfile, tile_bounds, res, srs, nodata)
        else:
            raise
    finally:
        if not keep_json:
            try:
                os.remove(jpath)
            except Exception:
                pass
        else:
            print(f"[DBG] Pipeline JSON ponechán: {jpath}")
    return outfile


def run_fillnodata(src, dst, maxdist, smooth):
    import os, shutil, sys

    candidates = [
        "gdal_fillnodata.exe",
        "gdal_fillnodata",
        "gdal_fillnodata.py",
    ]

    extra_dirs = []
    for root in (os.environ.get("OSGEO4W_ROOT"), r"D:\\OSGeo4W", r"C:\\OSGeo4W"):
        if root and os.path.isdir(root):
            extra_dirs += [
                os.path.join(root, r"apps\Python312\Scripts"),
                os.path.join(root, r"bin"),
            ]

    exe = None
    for c in candidates:
        p = shutil.which(c)
        if p:
            exe = [p]
            break
    if exe is None:
        for d in extra_dirs:
            for c in candidates:
                p = os.path.join(d, c)
                if os.path.exists(p):
                    exe = [p]
                    break
            if exe:
                break
    if exe is None:
        exe = [sys.executable, "-m", "osgeo_utils.gdal_fillnodata"]

    cmd = exe + [
        "-md", str(maxdist),
        "-si", str(smooth),
        "-of", "GTiff",
        "-co", "TILED=YES",
        "-co", "COMPRESS=DEFLATE",
        "-co", "PREDICTOR=2",
        "-co", "BIGTIFF=YES",
        src,
        dst,
    ]
    run(cmd, echo=True)


def main():
    ap = argparse.ArgumentParser(description="LAS/LAZ (soubor NEBO složka) -> GeoTIFF 1 m/px (max Z) – PDAL/GDAL CLI, paralelně, verbose")
    ap.add_argument("input_path", help="cesta k .las/.laz nebo složce s LAS/LAZ")
    ap.add_argument("output_tif")
    ap.add_argument("--resolution", "-r", type=float, default=1.0, help="pixel size [m]")
    ap.add_argument("--nodata", type=float, default=-9999.0, help="NoData value")
    ap.add_argument("--tile", type=float, default=1000.0, help="tile size [m]")
    ap.add_argument("--srs", default="EPSG:5514", help="override/output SRS")
    ap.add_argument("--jobs", "-j", type=int, default=os.cpu_count(), help="parallel processes")
    ap.add_argument("--keep-tiles", action="store_true", help="do not delete tiles after mosaic")
    ap.add_argument("--cog", action="store_true", help="write COG (needs gdal_translate)")
    ap.add_argument("--dtm", action="store_true", help="use only Classification==2 (ground)")
    ap.add_argument("--pdal-verbose", type=int, default=4, help="PDAL verbosity level (0–8)")
    ap.add_argument("--keep-json", action="store_true", help="keep per-tile PDAL pipeline JSON for debugging")
    ap.add_argument("--no-resume", action="store_true", help="NEpřeskakovat hotové dlaždice (vynutit přepočet)")
    ap.add_argument("--verify", action="store_true", help="při resume navíc ověřit dlaždice přes gdalinfo (pomalejší)")
    ap.add_argument("--fill", action="store_true", help="po mozaikování doplní NoData (gdal_fillnodata) a vytvoří *_filled.tif")
    ap.add_argument("--maxdist", type=float, default=50, help="max. vzdálenost interpolace v pixelech (gdal_fillnodata -md), výchozí 50")
    ap.add_argument("--smooth", type=int, default=2, help="počet vyhlazovacích iterací (gdal_fillnodata -si), výchozí 2")
    args = ap.parse_args()

    if not have("pdal"):
        die("Nenalezen 'pdal' v PATH.")
    if not have("gdalbuildvrt") or not have("gdal_translate"):
        die("Chybí 'gdalbuildvrt' nebo 'gdal_translate' v PATH.")

    infiles = list_las_inputs(args.input_path)
    if not infiles:
        die("Zadaný vstup není LAS/LAZ soubor ani složka s LAS/LAZ.")

    if len(infiles) == 1:
        src_desc = infiles[0]
    else:
        src_desc = f"{len(infiles)} souborů v '{os.path.abspath(args.input_path)}'"

    print("[INFO] Vstup:", src_desc)
    print("[INFO] Výstup:", args.output_tif)
    print("[INFO] Parametry: res=%.3f m, tile=%.1f m, jobs=%d, COG=%s, DTM=%s, resume=%s" % (
        args.resolution, args.tile, args.jobs, args.cog, args.dtm, not args.no_resume
    ))

    # bbox sjednoceně přes všechny vstupy
    minx, miny, maxx, maxy = get_bounds_many(infiles)
    print(f"[INFO] BBOX: min({minx:.3f},{miny:.3f}) max({maxx:.3f},{maxy:.3f})")

    # grid
    tx = math.ceil((maxx - minx) / args.tile)
    ty = math.ceil((maxy - miny) / args.tile)
    print(f"[INFO] Dlaždice: {tx} × {ty} = {tx*ty} ks")

    # root složka = složka výsledného TIFu
    out_base = os.path.abspath(os.path.dirname(args.output_tif))
    outdir = os.path.join(out_base, "tiles_tmp")
    os.makedirs(outdir, exist_ok=True)

    tasks = []
    tiles = []
    idx = 0
    classification = 2 if args.dtm else None

    for iy in range(ty):
        for ix in range(tx):
            bx0 = minx + ix * args.tile
            by0 = miny + iy * args.tile
            bx1 = min(bx0 + args.tile, maxx)
            by1 = min(by0 + args.tile, maxy)
            bounds = (bx0, by0, bx1, by1)
            outfile = tile_outpath(outdir, idx, bounds)

            if (not args.no_resume) and is_tile_done(outfile):
                if args.verify and shutil.which("gdalinfo"):
                    try:
                        subprocess.check_call(["gdalinfo", "-quiet", outfile], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        print(f"[SKIP] Už hotovo → {outfile}")
                        tiles.append(outfile)
                        idx += 1
                        continue
                    except subprocess.CalledProcessError:
                        print(f"[WARN] {outfile} neprošel ověřením → přepočítávám")
                else:
                    print(f"[SKIP] Už hotovo → {outfile}")
                    tiles.append(outfile)
                    idx += 1
                    continue

            tasks.append((
                infiles, bounds, args.resolution, args.nodata, args.srs, outdir, idx,
                classification, args.pdal_verbose, args.keep_json, not args.no_resume,
            ))
            idx += 1

    if tasks:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(worker, t) for t in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    out = f.result()
                    tiles.append(out)
                    print(f"[OK] Dlaždice {i}/{len(tasks)} hotová → {out}")
                except Exception as e:
                    print(f"[FAIL] Dlaždice {i}/{len(tasks)}: {e}", file=sys.stderr)
                    raise
    else:
        print("[INFO] Všechny dlaždice už existují – přeskočeno počítání.")

    if not tiles:
        die("Nejsou k dispozici žádné dlaždice pro mozaiku.")

    vrt = os.path.join(out_base, "tmp.vrt")
    tiles_sorted = sorted(tiles)
    print("[INFO] Kontroluji/opravuji datové typy dlaždic na Float32…")
    for p in tiles_sorted:
        ensure_float32(p, args.nodata)

    run(["gdalbuildvrt", "-overwrite", vrt] + tiles_sorted)

    if args.cog:
        run([
            "gdal_translate", vrt, args.output_tif, "-of", "COG",
            "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=2",
            "-co", "NUM_THREADS=ALL_CPUS", "-co", "BLOCKSIZE=512",
        ])
    else:
        run([
            "gdal_translate", vrt, args.output_tif,
            "-co", "TILED=YES", "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=2",
            "-co", "BIGTIFF=YES", "-co", "NUM_THREADS=ALL_CPUS",
            "-co", "BLOCKXSIZE=512", "-co", "BLOCKYSIZE=512",
        ])
    try:
        os.remove(vrt)
    except Exception:
        pass

    if args.fill:
        base, ext = os.path.splitext(args.output_tif)
        filled_gtiff = base + "_filled.tif"
        print("[INFO] Vyplňuji NoData (gdal_fillnodata)…")
        run_fillnodata(args.output_tif, filled_gtiff, args.maxdist, args.smooth)

        if args.cog:
            filled_cog = base + "_filled_cog.tif"
            run([
                "gdal_translate", filled_gtiff, filled_cog, "-of", "COG",
                "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=2",
                "-co", "NUM_THREADS=ALL_CPUS", "-co", "BLOCKSIZE=512",
            ])
            print(f"[INFO] Vyplněná verze (COG): {filled_cog}")
        else:
            print(f"[INFO] Vyplněná verze: {filled_gtiff}")

    if not args.keep_tiles:
        for p in tiles_sorted:
            try:
                os.remove(p)
            except Exception:
                pass

    print(f"[DONE] {os.path.abspath(args.output_tif)}")


if __name__ == "__main__":
    main()
