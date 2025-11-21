import time
from typing import Dict, Any
import logging

# Placeholder for influxdb_client
try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
except ImportError:
    InfluxDBClient = None
    Point = None
    SYNCHRONOUS = None

logger = logging.getLogger(__name__)

class DataIngestor:
    """
    Ingests market tick data into Time-Series Database (InfluxDB/QuestDB).
    """
    
    def __init__(self, url: str = "http://localhost:8086", token: str = "my-token", org: str = "my-org", bucket: str = "trading"):
        self.client = None
        self.write_api = None
        self.bucket = bucket
        self.org = org
        
        if InfluxDBClient:
            try:
                self.client = InfluxDBClient(url=url, token=token, org=org)
                self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
                logger.info("Connected to Time-Series DB")
            except Exception as e:
                logger.error(f"Failed to connect to TSDB: {e}")

    def write_tick_data(self, game_id: str, price: float, volume: float, features: Dict[str, Any]):
        """
        Write a single tick to the database.
        """
        if not self.write_api:
            return
            
        try:
            point = Point("nfl_ticks") \
                .tag("game_id", game_id) \
                .field("price", float(price)) \
                .field("volume", float(volume)) \
                .time(time.time_ns())
                
            # Add feature fields
            for k, v in features.items():
                if isinstance(v, (int, float)):
                    point.field(k, float(v))
                    
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            
        except Exception as e:
            logger.error(f"Failed to write tick: {e}")

    def close(self):
        if self.client:
            self.client.close()
