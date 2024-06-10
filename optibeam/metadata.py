from typing import *
from abc import ABC, abstractmethod
import json
import hashlib

class Metadata(ABC):
    @abstractmethod
    def add_metadata(self, key, value):
        pass
    
    @abstractmethod
    def get_hash(self):
        pass
    
    def to_sql_insert(self, table_name: str) -> str:
        # Extract column names and their corresponding values from metadata dictionary
        columns = ', '.join(self.metadata.keys())
        values = ', '.join([f"'{str(value)}'" if isinstance(value, str) else str(value) for value in self.metadata.values()])
        # Create the INSERT INTO statement
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"
        return sql

class ImageMetadata(Metadata):
    def __init__(self):
        self.metadata = {}
        
    def add_metadata(self, key, value):
        self.metadata[key] = value
        
    def get_hash(self):
        pass
        
    def set_image_metadata(self, meta: dict={}):
        self.metadata["image_id"] = meta['image_id'] 
        self.metadata["capture_time"] = meta['capture_time']
        self.metadata["original_crop_pos"] = meta['original_crop_pos']
        self.metadata["speckle_crop_pos"] = meta['speckle_crop_pos']
        self.metadata["beam_parameters"] = meta['beam_parameters']
        self.metadata["num_of_images"] = meta['num_of_images']
        self.metadata["metadata_id"] = meta['metadata_id']
        self.metadata["comments"] = meta['comments']

class ConfigMetaData(Metadata):
    def __init__(self):
        self.metadata = {}
        
    def add_metadata(self, key, value):
        self.metadata[key] = value
        
    def set_basic_experiment_metadata(self, experiment_name="", experiment_description="", experiment_location=""):
        self.metadata["experiment_name"] = experiment_name
        self.metadata["experiment_description"] = experiment_description
        self.metadata["experiment_location"] = experiment_location
        
    def set_config_metadata(self, fiber_config={}, camera_config={}, other_config={}):
        self.metadata["fiber_config"] = json.dumps(fiber_config)
        self.metadata["camera_config"] = json.dumps(camera_config)
        self.metadata["other_config"] = json.dumps(other_config)
        self._set_hash()
    
    def _set_hash(self):
        serialized_data = self.metadata["fiber_config"] + self.metadata["camera_config"] + self.metadata["other_config"]
        # Create a hash of the serialized string
        hash_object = hashlib.sha256(serialized_data.encode())
        self.metadata["hash"] = hash_object.hexdigest()
    
    def get_hash(self):
        return self.metadata["hash"]
    
