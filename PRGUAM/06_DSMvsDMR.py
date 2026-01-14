import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# --- Cesty k souborům ---
# dtm_path = r"G:\Python_env_PragUAM\DSM_DTM\Praha_5G_maxZ_1m_smoothed_filled.tif"
# dsm_path = r"G:\Python_env_PragUAM\DSM_DTM\Praha_1G_maxZ_1m.tif"
# out_path = r"G:\Python_env_PragUAM\DSM_DTM\Praha_DSM-DMR_computed.tif"


dtm_path = r"G:\Python_env_PragUAM\testing_location_01\data_dmr5g_testing_location_01\testing_location_01_5G_maxZ_1m_cog_filled_cog.tif"
dsm_path = r"G:\Python_env_PragUAM\testing_location_01\data_dmr1g_testing_location_01\testing_location_01_1G_maxZ_1m_cog.tif"
out_path = r"G:\Python_env_PragUAM\testing_location_01\testing_location_01_DSM-DMR_computed.tif"

# dtm_path = r"G:\Python_env_PragUAM\testing_location_03\testing_location_03_5G_maxZ_1m_cog_filled_cog.tif"
# dsm_path = r"G:\Python_env_PragUAM\testing_location_03\testing_location_03_1G_maxZ_1m_cog.tif"
# out_path = r"G:\Python_env_PragUAM\testing_location_03\testing_location_03_DSM-DMR_computed.tif"
# # --- PARAMETRY ---
nodata_val = -9999.0
resampling = Resampling.bilinear   # pro výšky OK; pro „tvrdé“ hranice lze nearest/cubic

# ---- 1) Načti referenční DTM (mřížka) ----
with rasterio.open(dtm_path) as dtm_ds:
    dtm = dtm_ds.read(1, masked=True)
    dtm_prof = dtm_ds.profile
    dtm_transform = dtm_ds.transform
    dtm_crs = dtm_ds.crs
    dtm_nodata_src = dtm_ds.nodata

# ---- 2) Načti DSM a zarovnej na mřížku DTM ----
with rasterio.open(dsm_path) as dsm_ds:
    dsm_src = dsm_ds.read(1)  # ndarray
    dsm_src_nodata = dsm_ds.nodata

    dsm_aligned = np.empty(dtm.shape, dtype=np.float32)
    dsm_aligned.fill(nodata_val)

    reproject(
        source=dsm_src,
        destination=dsm_aligned,
        src_transform=dsm_ds.transform,
        src_crs=dsm_ds.crs,
        src_nodata=dsm_src_nodata,
        dst_transform=dtm_transform,
        dst_crs=dtm_crs,
        dst_nodata=nodata_val,
        resampling=resampling
    )

# ---- 3) Vytvoř masked arrays (DSM/DTM) ----
dsm_ma = np.ma.masked_where(dsm_aligned == nodata_val, dsm_aligned)

# DTM maska: respektuj původní NoData; když není, maskuj jen NaN
if dtm_prof.get("nodata") is not None:
    dtm_nodata = dtm_prof["nodata"]
elif dtm_nodata_src is not None:
    dtm_nodata = dtm_nodata_src
else:
    dtm_nodata = nodata_val

dtm_ma = np.ma.masked_where(dtm.filled(dtm_nodata) == dtm_nodata, dtm)

dsm_mask = np.ma.getmaskarray(dsm_ma)
dtm_mask = np.ma.getmaskarray(dtm_ma)

# ---- 4) Rozdíl dle pravidel:
#      - chybí DSM -> 0
#      - chybí DTM -> NoData
#      - jinak     -> DSM - DTM
diff = np.empty(dtm_ma.shape, dtype=np.float32)

# kde chybí DTM -> NoData
diff[:] = nodata_val
# kde chybí DSM (a zároveň máme DTM) -> 0
only_dsm_missing = dsm_mask & (~dtm_mask)
diff[only_dsm_missing] = 0.0
# kde jsou obě data -> skutečný rozdíl
both_ok = (~dsm_mask) & (~dtm_mask)
diff[both_ok] = (dsm_ma.filled(0) - dtm_ma.filled(0))[both_ok]

# ---- 5) Zápis 3-band GeoTIFFu s popisy bandů ----
out_prof = dtm_prof.copy()
out_prof.update(
    count=3,
    dtype="float32",
    nodata=nodata_val,
    tiled=True,
    compress="DEFLATE",
    predictor=2,
    BIGTIFF="YES"
)

with rasterio.open(out_path, "w", **out_prof) as dst:
    # Band 1: DSM (NoData kde chybí DSM)
    dst.write(dsm_ma.filled(nodata_val).astype("float32"), 1)
    dst.set_band_description(1, "DSM_height")

    # Band 2: DTM (NoData kde chybí DTM)
    dst.write(dtm_ma.filled(nodata_val).astype("float32"), 2)
    dst.set_band_description(2, "DTM_height")

    # Band 3: rozdíl dle pravidel výše
    dst.write(diff.astype("float32"), 3)
    dst.set_band_description(3, "Height_diff_DSM_minus_DTM")


print(f"✅ Hotovo → {out_path}")
