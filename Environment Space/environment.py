import ee

# Initialize Earth Engine
ee.Initialize(project='pipeline-digital-twin-490415')

print("Connected to Earth Engine")

# ---------------------------
# 1. TEXAS BOUNDARY
# ---------------------------

states = ee.FeatureCollection("TIGER/2018/States")

texas = states.filter(ee.Filter.eq('NAME', 'Texas'))

print("Texas boundary loaded")

# ---------------------------
# 2. PIPELINE DATA (OGIM)
# ---------------------------

ogim = ee.FeatureCollection("EDF/OGIM/current")

# Keep only infrastructure inside Texas
tx_pipelines = ogim.filterBounds(texas)

# Optional: limit dataset for faster export
tx_pipelines = tx_pipelines.limit(1000)

print("Texas pipelines loaded")

# ---------------------------
# 3. ENVIRONMENT DATA
# ---------------------------

# Soil pH (corrosion factor)
soil_ph = ee.Image(
    "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02"
).select('b0').rename('soil_pH')

# Clay content (soil chemistry)
clay = ee.Image(
    "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02"
).select('b0').rename('clay')

# Elevation
dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation')

# Slope from DEM
slope = ee.Terrain.slope(dem).rename('slope')

# Clip everything to Texas
soil_ph = soil_ph.clip(texas)
clay = clay.clip(texas)
dem = dem.clip(texas)
slope = slope.clip(texas)

print("Environmental layers prepared")

# ---------------------------
# 4. STACK ENVIRONMENT DATA
# ---------------------------

env = ee.Image.cat([
    soil_ph,
    clay,
    dem,
    slope
])

print("Environment stack ready")

# ---------------------------
# 5. SAMPLE ENVIRONMENT
# ---------------------------

samples = env.reduceRegions(
    collection=tx_pipelines,
    reducer=ee.Reducer.mean(),
    scale=250
)

print("Sampling environmental variables")

# ---------------------------
# 6. EXPORT DATASET
# ---------------------------

task = ee.batch.Export.table.toDrive(
    collection=samples,
    description='Texas_Pipeline_Environment',
    fileFormat='CSV',
    folder='Digital_Twin_Project'
)

task.start()

print("Export started — check Google Drive soon.")