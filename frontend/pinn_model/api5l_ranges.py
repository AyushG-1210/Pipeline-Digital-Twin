"""
api5l_ranges.py
================
Single source of truth for API 5L physical parameter ranges and branch
input schemas used across the Pipeline Digital Twin PINN.

API 5L Standard References
---------------------------
* API Spec 5L, 46th Edition (2018) -- base standard for seamless & welded line pipe.
* PHMSA HL Annual Report Part H -- OD bins: 4-less, 6, 8 ... 46, 48, 52, 56, 58+over.
* amerpipe.com/products/api-5l-pipe-specifications/ -- Sizes summary:
    Seamless: 0.5" NPS to 36" OD
    ERW:      2" NPS to 26" OD
    SAW:      20" to 60" OD

Verification date: 2026-09-13
"""

# =============================================================================
# WALL THICKNESS (WT) -- inches
# =============================================================================
# Lower bound: 0.188" corresponds to ASME B36.10 Schedule 20 at 4" NPS,
#   which is the smallest diameter in scope (PHMSA Part H lower bin = "4-less").
#   Thinner than 0.188" falls into gathering/distribution, not transmission.
# Upper bound: 1.500" covers heavy-wall X-grade transmission pipe up to ~36" OD
#   at high SMYS grades (X65-X80). Specialty heavy-wall > 2" is out of scope.
WT_MIN_IN = 0.188   # inches -- smallest in-scope transmission wall thickness
WT_MAX_IN = 1.500   # inches -- largest in-scope transmission wall thickness

# Equivalents in meters (used by the FDM solver: L = WT_inches * 0.0254)
WT_MIN_M = WT_MIN_IN * 0.0254   # 0.004775 m
WT_MAX_M = WT_MAX_IN * 0.0254   # 0.03810  m

# =============================================================================
# OUTER DIAMETER (OD) -- inches
# =============================================================================
# Lower bound: 4.0" -- PHMSA HL Part H lowest reportable bin ("4-less" means
#   pipe <= 4" OD; we set the model minimum at 4.0" as the boundary).
# Upper bound: 56.0" -- the last *full* PHMSA bin before the open-ended
#   "58+ over" bin. 58" cannot anchor a normalization max because it has no
#   upper limit; 56" is the largest safely bounded diameter in the data.
#   SAW pipe physically extends to 60", but > 56" is not reportable as a
#   distinct PHMSA bin and is extremely rare in Texas transmission.
OD_MIN_IN = 4.0    # inches
OD_MAX_IN = 56.0   # inches

# =============================================================================
# META TENSOR / RAW TENSOR -- column index schema
# =============================================================================
# Both meta_tensor (normalized) and raw_tensor (physical) are shaped [B, 3].
# Column layout is identical; only the value scale differs.
#
#   meta_tensor[:, META_COL_WT]   = (wt_raw - WT_MIN_IN) / (WT_MAX_IN - WT_MIN_IN)  in [0, 1]
#   meta_tensor[:, META_COL_OD]   = (od_raw - OD_MIN_IN) / (OD_MAX_IN - OD_MIN_IN)  in [0, 1]
#   meta_tensor[:, META_COL_YOLO] = YOLO defect confidence score                     in [0, 1]
#
#   raw_tensor[:, META_COL_WT]    = wt_raw  in inches, range [WT_MIN_IN, WT_MAX_IN]
#   raw_tensor[:, META_COL_OD]    = od_raw  in inches, range [OD_MIN_IN, OD_MAX_IN]
#   raw_tensor[:, META_COL_YOLO]  = same as meta (already unit-less)
META_COL_WT   = 0   # wall thickness column index
META_COL_OD   = 1   # outer diameter column index
META_COL_YOLO = 2   # YOLO defect score column index

META_TENSOR_NCOLS = 3
META_TENSOR_DTYPE = "float32"

# =============================================================================
# BRANCH 1 (SOIL) -- real-world column schema
# =============================================================================
# Source: Neo4j graph DB, enriched by:
#   - PHMSA incident GPS coordinates (lat/lon)
#   - ERA5-Land MONTHLY_AGGR via Google Earth Engine (soil_moisture, temperature)
#     Script: GeEarthConnectivity/gee_climate.py
#
# Tensor shape when using REAL data:      [N_incidents, 4]  (N_incidents = 5,831 at Phase 4)
# Tensor shape when using SYNTHETIC data: [Batch, SOIL_POINTS]  (SOIL_POINTS = 8, GRF profiles)
#
# Col | Field         | Source band                        | Units    | Texas range
# ----|---------------|------------------------------------|----------|-------------------
#  0  | lat           | Incident GPS                       | deg (N)  | 25.8  to  36.5
#  1  | lon           | Incident GPS                       | deg (E)  | -106.6 to -93.5
#  2  | soil_moisture | ERA5 volumetric_soil_water_layer_1 | m3/m3    | 0.05 to 0.55
#  3  | temperature   | ERA5 temperature_2m  (K -> C)      | degC     | -5.0 to 45.0

SOIL_COL_LAT         = 0
SOIL_COL_LON         = 1
SOIL_COL_MOISTURE    = 2
SOIL_COL_TEMPERATURE = 3

SOIL_REAL_NCOLS = 4   # used when DATA_MODE == DATA_MODE_REAL

# Normalization bounds for [0, 1] rescaling before model input.
# Clamp raw values to these bounds before applying (x - lo) / (hi - lo).
SOIL_NORM_BOUNDS = {
    "lat":          (25.8,   36.5),    # Texas N-S extent (decimal degrees)
    "lon":          (-106.6, -93.5),   # Texas W-E extent (decimal degrees)
    "soil_moisture":(0.05,   0.55),    # m3/m3 -- dry caliche to saturated clay
    "temperature":  (-5.0,   45.0),    # degC  -- winter low to summer peak (2m air)
}

# =============================================================================
# BRANCH 2 (FLUID) -- synthetic declaration (Phase 4 / 5 / 6)
# =============================================================================
# STATUS: SYNTHETIC -- no per-segment operating pressure or fluid flow rate
# exists in any currently ingested dataset.
#
# PHMSA HL Annual Report audit (conducted 2026-09-13):
#   Part A-E : Commodity type, operator metadata          -- NO pressure/flow
#   Part F-G : Mileage by pipe condition / HCA status     -- NO pressure/flow
#   Part H   : Mileage binned by OD (4" to 58"+)         -- NO pressure/flow
#   Part J   : Mileage by seam type x operating stress   -- PARTIAL SIGNAL ONLY
#              (PARTJST20MOREON etc. = miles at >=20% SMYS,
#               system-level aggregate, NOT per-segment values)
#   RRC GIS  : Static permit data (T-4)                   -- NO pressure/flow
#   RRC Datasets : Annual mileage statistics              -- NO pressure/flow
#
# FUTURE ROADMAP: Part J %SMYS bands could constrain fluid branch distribution
# (e.g., pipe with >60% of miles at >=20% SMYS implies higher typical operating
# pressure). Proposed as a Phase 7+ enhancement.
#
# CURRENT IMPLEMENTATION: Branch 2 uses GRF profiles (50 nodes) sampled
# uniformly in [0,1], mapped to C_bulk in [0.3, 1.0] via fluid_to_cbulk().
FLUID_BRANCH_IS_SYNTHETIC = True
FLUID_BRANCH_NCOLS        = 50     # GRF profile nodes (FLUID_POINTS in main.ipynb)
C_BULK_MIN                = 0.3    # minimum local O2 concentration (normalized)
C_BULK_MAX                = 1.0    # maximum local O2 concentration (normalized)

# =============================================================================
# DATA MODE FLAG
# =============================================================================
# "synthetic" : Use GRF-generated soil profiles (soil_dim = SOIL_POINTS = 8)
#               and GRF fluid profiles (fluid_dim = FLUID_POINTS = 50).
#               Active path for PINN training in main.ipynb.
# "real"      : Use Neo4j/GEE exported input_tensor.npy (soil_dim = 4).
#               Active path for inference/evaluation against real incidents.
#               FactoredMIONet must be instantiated with soil_dim=SOIL_REAL_NCOLS.
DATA_MODE_SYNTHETIC = "synthetic"
DATA_MODE_REAL      = "real"
