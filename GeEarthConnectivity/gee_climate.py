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
    # Use ERA5-Land for a stable, globally complete dataset
    # This dataset contains both soil moisture and temperature in one place.
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate('2023-01-01', '2023-12-31').mean()
    
    # Select the specific bands
    moisture_band = era5.select('volumetric_soil_water_layer_1')
    temp_band = era5.select('temperature_2m')
    
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            print("Fetching incidents from Neo4j...")
            incidents = session.execute_read(fetch_incidents)
            
            if not incidents:
                print("No incidents found needing climate data.")
                return

            print(f"Fetching soil moisture and temperature for {len(incidents)} incidents...")
            for idx, incident in enumerate(incidents):
                try:
                    point = ee.Geometry.Point([incident['lon'], incident['lat']])
                    
                    # Sample the data 
                    moisture_data = moisture_band.sample(point, 1000).first().getInfo()
                    temp_data = temp_band.sample(point, 1000).first().getInfo()
                    
                    # Extract values (default to 0.0 if no data is found)
                    m_val = moisture_data['properties']['volumetric_soil_water_layer_1'] if moisture_data else 0.0
                    
                    # Extract temperature and convert from Kelvin to Celsius
                    t_raw = temp_data['properties']['temperature_2m'] if temp_data else 0.0
                    t_val = (t_raw - 273.15) if t_raw > 0 else 0.0
                    
                    # Write back to graph
                    session.execute_write(update_incident_climate, incident['report_number'], m_val, t_val)
                    
                    # Print progress every 100 records to keep the terminal clean
                    if (idx + 1) % 100 == 0 or (idx + 1) == len(incidents):
                        print(f"[{idx+1}/{len(incidents)}] Processed Incident {incident['report_number']} | Moisture: {m_val:.3f} | Temp: {t_val:.2f}°C")
                    
                except Exception as e:
                    print(f"Failed to process incident {incident['report_number']}: {e}")

    print("Task 2: GEE API Scraping (Soil Moisture & Temperature) Complete!")

if __name__ == "__main__":
    fetch_climate_data()
