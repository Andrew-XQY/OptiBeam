import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import ast

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



# train_batch = ['train']
# valid_batch = ['valid']
# test_batch = ['test']
# dim = (256, 256)
# shape = (None, 256, 256, 1)
# batch_size = 32

# datasets = []

# for batch in [train_batch, valid_batch, test_batch]:
#     datasets.append(create_tf_dataset(df, batch, dim, shape))

# train, valid, test = datasets
