import os
import ee
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load variables from the .env file
load_dotenv()

# 1. Neo4j Credentials
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))

# 2. Initialize Earth Engine
print("Initializing Google Earth Engine...")
try:
    ee.Initialize(project=os.getenv("GEE_PROJECT")) 
except Exception as e:
    print("GEE Initialization failed.")
    raise e

def fetch_incidents(tx):
    """Pulls incidents that need soil moisture and temperature."""
    query = """
    MATCH (i:Incident)
    WHERE i.latitude IS NOT NULL 
      AND i.longitude IS NOT NULL 
      AND i.soil_moisture IS NULL
    RETURN i.report_number AS report_number, i.latitude AS lat, i.longitude AS lon
    LIMIT 100 // Remove or increase this limit for the full dataset later
    """
    result = tx.run(query)
    return [record.data() for record in result]

def update_incident_climate(tx, report_number, moisture, temp):
    """Writes the temperature and soil moisture back to the graph."""
    query = """
    MATCH (i:Incident {report_number: $report_number})
    SET i.soil_moisture = $moisture,
        i.temperature = $temp
    """
    tx.run(query, report_number=report_number, moisture=moisture, temp=temp)

def fetch_climate_data():
    # Load long-term averages to ensure we get a reading regardless of cloud cover
    # 1. Soil Moisture (NASA SMAP)
    smap = ee.ImageCollection('NASA_USDA/SMAP_SM400D_v2_0').mean()
    moisture_band = smap.select('ssm') # Surface soil moisture
    
    # 2. Temperature (MODIS Land Surface Temp)
    modis = ee.ImageCollection('MODIS/061/MOD11A1').mean()
    # MODIS temp is in Kelvin scaled by 0.02. We'll grab the raw value first.
    temp_band = modis.select('LST_Day_1km') 
    
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            incidents = session.execute_read(fetch_incidents)
            
            if not incidents:
                print("No incidents found needing climate data.")
                return

            print(f"Fetching climate data for {len(incidents)} incidents...")
            for idx, incident in enumerate(incidents):
                try:
                    point = ee.Geometry.Point([incident['lon'], incident['lat']])
                    
                    # Sample the data
                    moisture_data = moisture_band.sample(point, 30).first().getInfo()
                    temp_data = temp_band.sample(point, 30).first().getInfo()
                    
                    # Extract values (default to 0 if no data is found at that coordinate)
                    m_val = moisture_data['properties']['ssm'] if moisture_data else 0
                    
                    # Convert MODIS temp to Celsius ((Kelvin * 0.02) - 273.15)
                    t_raw = temp_data['properties']['LST_Day_1km'] if temp_data else 0
                    t_val = (t_raw * 0.02) - 273.15 if t_raw > 0 else 0
                    
                    session.execute_write(update_incident_climate, incident['report_number'], m_val, t_val)
                    print(f"[{idx+1}/{len(incidents)}] Incident {incident['report_number']} | Moisture: {m_val:.2f} | Temp: {t_val:.2f}°C")
                    
                except Exception as e:
                    print(f"Failed to process incident {incident['report_number']}: {e}")

    print("Task 2: Climate Enrichment Complete!")

if __name__ == "__main__":
    fetch_climate_data()
