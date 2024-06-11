from typing import *
from abc import ABC, abstractmethod
import json
import hashlib

class Metadata(ABC):
    def __init__(self):
        self.metadata = {}
        
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
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({values})" # Create the INSERT INTO statement
        return sql

class ImageMetadata(Metadata):
    def __init__(self):
        super().__init__()
        
    def add_metadata(self, key, value):
        self.metadata[key] = value
        
    def get_hash(self):
        temp_metadata = {key: value for key, value in self.metadata.items() if key != 'hash'}
        serialized_data = json.dumps(temp_metadata, sort_keys=True)
        hash_object = hashlib.sha512(serialized_data.encode())
        return hash_object.hexdigest()
        
    def set_image_metadata(self, meta: dict={}):
        for key, value in meta.items():
            self.metadata[key] = value if type(value) is not dict else json.dumps(value)

class ConfigMetaData(Metadata):
    def __init__(self):
        super().__init__()
        
    def add_metadata(self, key, value):
        self.metadata[key] = value
        
    def set_config_metadata(self, meta: dict={}):
        for key, value in meta.items():
            self.metadata[key] = value if type(value) is not dict else json.dumps(value)
        self._set_hash()
    
    def _set_hash(self):
        temp_metadata = {key: value for key, value in self.metadata.items() if key != 'hash'}
        serialized_data = json.dumps(temp_metadata, sort_keys=True)
        hash_object = hashlib.sha512(serialized_data.encode())
        self.metadata["hash"] = hash_object.hexdigest()
    
    def get_hash(self):
        return self.metadata["hash"]
    
