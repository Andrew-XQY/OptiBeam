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
        metadata = {key: value for key, value in self.metadata.items() if value is not None}
        columns = ', '.join(metadata.keys())
        values = ', '.join([f"'{str(value)}'" if isinstance(value, str) else str(value) for value in metadata.values()])
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
        self.metadata["image_id"] = meta.get('image_id')
        self.metadata["capture_time"] = meta.get('capture_time')
        self.metadata["original_crop_pos"] = meta.get('original_crop_pos')
        self.metadata["speckle_crop_pos"] = meta.get('speckle_crop_pos')
        self.metadata["beam_parameters"] = meta.get('beam_parameters')
        self.metadata["num_of_images"] = meta.get('num_of_images')
        self.metadata["image_path"] = meta.get('image_path')
        self.metadata["metadata_id"] = meta.get('metadata_id')
        self.metadata["batch"] = meta.get('batch')
        self.metadata["comments"] = meta.get('comments')

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
        temp_metadata = {key: value for key, value in self.metadata.items() if key != 'hash'}
        serialized_data = json.dumps(temp_metadata, sort_keys=True)
        hash_object = hashlib.sha512(serialized_data.encode())
        self.metadata["hash"] = hash_object.hexdigest()
    
    def get_hash(self):
        return self.metadata["hash"]
    
