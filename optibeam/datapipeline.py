import tensorflow as tf
import numpy as np
import pandas as pd
import ast

from PIL import Image
from abc import ABC, abstractmethod
from typing import *
from .utils import get_all_file_paths
from .database import Database


# ----------------- new tf pipeline with prefetch ----------------- 

class DataLoader(ABC):
    def __init__(self, df: pd.DataFrame=None):
        self.df = df
        self.batch_size = None
        self.shape = None
        
    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        
    def set_shape(self, shape: Tuple[int, int]):
        self.shape = shape
        
    def set_dataframe(self, df: pd.DataFrame):
        self.df = df
        
    @abstractmethod
    def get_metadata_from_db(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def data_pipeline(self, batch_list: List[int], dim: Tuple[int, int], is_batch: bool) -> tf.data.Dataset:
        pass

    def create_tf_dataset(self, batch_list: List[int], dim: Tuple[int, int], is_batch: bool) -> tf.data.Dataset:
        return tf.data.Dataset.from_generator(
            generator=lambda: self.data_pipeline(batch_list, dim, is_batch),
            output_types=(tf.float32, tf.float32),
            output_shapes=(self.shape, self.shape)
        ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)



class TF_DataLoader(DataLoader):
    def __init__(self, db: Database):
        super().__init__(db)
        
    def get_metadata_from_db(self) -> pd.DataFrame:
        return self.db.get_metadata()
    
    def data_pipeline(self, batch_list: List[int], dim: Tuple[int, int], is_batch: bool) -> tf.data.Dataset:
        return tf.data.Dataset.from_generator(
            generator=lambda: self.data_pipeline_generator(batch_list, dim, is_batch),
            output_types=(tf.float32, tf.float32),
            output_shapes=(self.shape, self.shape)
        ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    def data_pipeline_generator(self, batch_list: List[int], dim: Tuple[int, int], is_batch: bool):
        batch_x, batch_y = [], []
        metadata = self.get_metadata_from_db()
        while True:  # Loop indefinitely
            for index, row in metadata.iterrows():
                img = Image.open(row['image_path']).convert('L')  # Convert to grayscale
                crop_x = ast.literal_eval(row["speckle_crop_pos"])
                crop_y = ast.literal_eval(row["original_crop_pos"])
                crop_x = tuple(item for subtuple in crop_x for item in subtuple)
                crop_y = tuple(item for subtuple in crop_y for item in subtuple)
                img_x = img.crop(crop_x)  # crop ROI
                img_y = img.crop(crop_y)
                img_x = img_x.resize(dim)   # Resize dimensions
                img_y = img_y.resize(dim)
                res_x = np.expand_dims(np.array(img_x), axis=-1) # Change shape to (256, 256, 1)
                res_y = np.expand_dims(np.array(img_y), axis=-1)
                if is_batch:
                    batch_x.append(np.array(res_x)) 
                    batch_y.append(np.array(res_y)) 
                    if len(batch_x) >= self.batch_size:  # Yield a batch when batch size is reached
                        batch_x = np.stack(batch_x)
                        batch_y = np.stack(batch_y)
                        yield batch_x.astype('float32') / 255., batch_y.astype('float32') / 255.
                        batch_x, batch_y = [], []
                else:
                    yield res_x.astype('float32') / 255., res_y.astype('float32') / 255.










# ----------------- old data pipeline ----------------- 
class DataPipeline:
    def __init__(self, df, shape):
        self.df = df
        self.shape = shape
    
    def data_pipeline(self, dim, batch_size=1, is_batch=True):
        batch_x, batch_y = [], []
        while True:  # Loop indefinitely
            for index, row in self.df.iterrows():
                img = Image.open(row['image_path']).convert('L')  # Convert to grayscale
                crop_x = ast.literal_eval(row["speckle_crop_pos"])
                crop_y = ast.literal_eval(row["original_crop_pos"])
                crop_x = tuple(item for subtuple in crop_x for item in subtuple)
                crop_y = tuple(item for subtuple in crop_y for item in subtuple)
                img_x = img.crop(crop_x)  # crop ROI
                img_y = img.crop(crop_y)
                img_x = img_x.resize(dim)   # Resize dimensions
                img_y = img_y.resize(dim)
                res_x = np.expand_dims(np.array(img_x), axis=-1) # Change shape to (256, 256, 1)
                res_y = np.expand_dims(np.array(img_y), axis=-1)
                if is_batch:
                    batch_x.append(np.array(res_x)) 
                    batch_y.append(np.array(res_y)) 
                    if len(batch_x) >= batch_size:  # Yield a batch when batch size is reached
                        batch_x = np.stack(batch_x)
                        batch_y = np.stack(batch_y)
                        yield batch_x.astype('float32') / 255., batch_y.astype('float32') / 255.
                        batch_x, batch_y = [], []
                else:
                    yield res_x.astype('float32') / 255., res_y.astype('float32') / 255.

    def create_tf_dataset(self, batch_list, dim=(256, 256), batch_size=1, is_batch=True):
        return tf.data.Dataset.from_generator(
            generator=lambda: self.data_pipeline(df=self.df[self.df['batch'].isin(batch_list)], dim=dim, batch_size=batch_size),
            output_types=(tf.float32, tf.float32),
            output_shapes=(self.shape, self.shape)
        ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
