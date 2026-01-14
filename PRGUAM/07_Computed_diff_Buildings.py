import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.crs import CRS
from rasterio.errors import DriverRegistrationError

# === Uprav cesty ===
# PATH_DIFF = r"G:\Python_env_PragUAM\DSM_DTM\Praha_DSM-DMR_computed_COG.tif"   # 3-band: [DSM, DTM, DSM-DTM]
# PATH_REL  = r"G:\Python_env_PragUAM\DSM_DTM\bud_rel.tif"                      # 1-band: relativní výšky budov (0 = žádná budova)
# PATH_OUT  = r"G:\Python_env_PragUAM\DSM_DTM\DSM_DTM_with_building_band.tif"   # výstupní COG GeoTIFF

PATH_DIFF = r"G:\Python_env_PragUAM\testing_location_01\testing_location_01_DSM-DMR_computed.tif"   # 3-band: [DSM, DTM, DSM-DTM]
PATH_REL  = r"G:\Python_env_PragUAM\final_scripts\bud_rel.tif"                      # 1-band: relativní výšky budov (0 = žádná budova)
PATH_OUT  = r"G:\Python_env_PragUAM\testing_location_01\testing_location_01_DSM_DTM_with_building_band.tif"  # výstupní COG GeoTIFF



# PATH_DIFF = r"G:\Python_env_PragUAM\testing_location_03\testing_location_03_DSM-DMR_computed.tif"   # 3-band: [DSM, DTM, DSM-DTM]
# PATH_REL  = r"G:\Python_env_PragUAM\DSM_DTM\bud_rel.tif"                      # 1-band: relativní výšky budov (0 = žádná budova)
# PATH_OUT  = r"G:\Python_env_PragUAM\testing_location_03\testing_location_03_DSM_DTM_with_building_band.tif"  # výstupní COG GeoTIFF


# --- Pomocné funkce ---
def read_band(src, bidx=1, masked=True, dtype=np.float32):
    return src.read(bidx, masked=masked).astype(dtype, copy=False)

def _normalize_crs(crs):
    """
    Vrať spolehlivý EPSG CRS, kdykoliv to dává smysl.
    Některé GeoTIFFy mají S-JTSK/Křovák jako EngineeringCRS s id EPSG:5514,
    což PROJ neumí transformovat na 'EPSG:5514'. Tady to sjednotíme.
    """
    c = CRS.from_user_input(crs)

    # 1) standardní cesta
    try:
        epsg = c.to_epsg()
        if epsg:
            return CRS.from_epsg(epsg)
    except Exception:
        pass

    # 2) authority (když to_epsg selže, ale authority je k dispozici)
    try:
        auth = c.to_authority()
        if auth and auth[0].upper() == "EPSG" and auth[1]:
            return CRS.from_epsg(int(auth[1]))
    except Exception:
        pass

    # 3) hrubá heuristika přes textovou reprezentaci
    s = c.to_string().upper()
    if "EPSG" in s and "5514" in s:
        return CRS.from_epsg(5514)

    return c


def reproject_to_match(src_from, arr_from, src_to,
                       resampling=Resampling.nearest,
                       treat_zero_as_valid=True,
                       dtype=np.float32):
    """
    Zarovná arr_from na grid src_to.
    - treat_zero_as_valid=True: 0 je platná hodnota (např. REL = bez budovy),
      takže nepoužijeme 0 jako NoData.
    - Normalizuje CRSy, aby se předešlo chybě 'Cannot find coordinate operations ...'.
    """
    # připrav zdrojové pole + src_nodata
    if isinstance(arr_from, np.ma.MaskedArray):
        if treat_zero_as_valid:
            src_array = arr_from.filled(0).astype(dtype, copy=False)
            src_nodata = None
        else:
            src_array = arr_from.filled(np.nan).astype(dtype, copy=False)
            src_nodata = float('nan')
    else:
        src_array = np.asarray(arr_from, dtype=dtype)
        nd = src_from.nodata
        if isinstance(nd, (list, tuple, np.ndarray)):
            nd = nd[0]
        if treat_zero_as_valid and nd == 0:
            nd = None
        src_nodata = nd

    src_crs_norm = _normalize_crs(src_from.crs)
    dst_crs_norm = _normalize_crs(src_to.crs)

    # Pokud obě strany „vypadají“ jako 5514 (i když jedna je EngineeringCRS),
    # vynutíme stejný EPSG objekt, aby PROJ neřešil rozdílný typ.
    if ("5514" in src_crs_norm.to_string() and "5514" in dst_crs_norm.to_string()):
        src_crs_norm = CRS.from_epsg(5514)
        dst_crs_norm = CRS.from_epsg(5514)
        
    print("REL CRS:", rel_src.crs)
    print("DIFF CRS:", diff_src.crs)
    print("REL CRS str:", rel_src.crs.to_string())
    print("DIFF CRS str:", diff_src.crs.to_string())

    out = np.empty((src_to.height, src_to.width), dtype=dtype)

    reproject(
        source=src_array,
        destination=out,
        src_transform=src_from.transform,
        src_crs=src_crs_norm,
        dst_transform=src_to.transform,
        dst_crs=dst_crs_norm,
        src_nodata=src_nodata,
        dst_nodata=None,
        resampling=resampling,
    )
    return out

# --- Hlavní výpočet ---
with rasterio.open(PATH_DIFF) as diff_src, rasterio.open(PATH_REL) as rel_src:
    # 1) načtení vstupů
    DSM = read_band(diff_src, 1)  # DSM
    DTM = read_band(diff_src, 2)  # DTM
    DIFF = read_band(diff_src, 3) # DSM - DTM
    REL = read_band(rel_src, 1)   # relativní výšky (0 = žádná budova)

    # 2) jestli je nutné zarovnání
    same_grid = (
        _normalize_crs(diff_src.crs) == _normalize_crs(rel_src.crs) and
        diff_src.transform == rel_src.transform and
        diff_src.width == rel_src.width and
        diff_src.height == rel_src.height
    )
    if not same_grid:
        REL_aligned = reproject_to_match(
            rel_src, REL, diff_src,
            resampling=Resampling.nearest,   # aby 0 zůstávalo 0
            treat_zero_as_valid=True,
            dtype=np.float32
        )
    else:
        REL_aligned = np.ma.filled(REL, 0).astype(np.float32, copy=False)

    # 3) příprava polí
    rel_data = np.asarray(REL_aligned, dtype=np.float32)               # 0 = bez budovy
    diff_data = np.ma.filled(DIFF, np.nan).astype(np.float32, copy=False)  # NaN = bez info

    # 4) nový band „Height_diff_building“
    new_band = np.where(rel_data > 0.0, rel_data, diff_data)
    mask_no_bldg = rel_data <= 0.0
    mask_neg = np.isfinite(new_band) & (new_band < 0.0)
    new_band[mask_no_bldg & mask_neg] = 0.0  # záporné mimo budovy sraz na 0

    # 5) zápis (COG)
    profile = diff_src.profile.copy()
    profile.update(
        count=4,
        dtype=np.float32,
        nodata=None,
    )

    out_b1 = (DSM.filled(np.nan) if isinstance(DSM, np.ma.MaskedArray) else DSM).astype(np.float32, copy=False)
    out_b2 = (DTM.filled(np.nan) if isinstance(DTM, np.ma.MaskedArray) else DTM).astype(np.float32, copy=False)
    out_b3 = (DIFF.filled(np.nan) if isinstance(DIFF, np.ma.MaskedArray) else DIFF).astype(np.float32, copy=False)
    out_b4 = new_band.astype(np.float32, copy=False)

    def write_as_cog(dst_path):
        prof = profile.copy()
        prof.update(
            driver="COG",
            COMPRESS="DEFLATE",
            BIGTIFF="IF_SAFER",
            BLOCKSIZE=512,
            NUM_THREADS="ALL_CPUS",
            RESAMPLING="NEAREST",
            OVERVIEWS="AUTO",
        )
        with rasterio.open(dst_path, "w", **prof) as dst:
            dst.write(out_b1, 1); dst.set_band_description(1, "DSM")
            dst.write(out_b2, 2); dst.set_band_description(2, "DTM")
            dst.write(out_b3, 3); dst.set_band_description(3, "Height_diff_DSM_minus_DTM")
            dst.write(out_b4, 4); dst.set_band_description(4, "Height_diff_building")

    def write_as_gtiff(dst_path):
        prof = profile.copy()
        prof.update(
            driver="GTiff",
            tiled=True,
            compress="DEFLATE",
            predictor=2,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(dst_path, "w", **prof) as dst:
            dst.write(out_b1, 1); dst.set_band_description(1, "DSM")
            dst.write(out_b2, 2); dst.set_band_description(2, "DTM")
            dst.write(out_b3, 3); dst.set_band_description(3, "Height_diff_DSM_minus_DTM")
            dst.write(out_b4, 4); dst.set_band_description(4, "Height_diff_building")

    try:
        write_as_cog(PATH_OUT)
    except (DriverRegistrationError, rasterio.errors.RasterioIOError, ValueError):
        # fallback na GTiff, když COG není dostupný nebo odmítne volby
        write_as_gtiff(PATH_OUT)

print("✅ Hotovo:", PATH_OUT)
print("Pozn.: CRS normalizováno na EPSG tam, kde to šlo (řeší PROJ chybu).")
