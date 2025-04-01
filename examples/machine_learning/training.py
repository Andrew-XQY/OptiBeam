# ============================
# Initialization
# ============================

import os
script_path = os.path.abspath(__file__)  # Get the absolute path of the current .py file
up_two_levels = os.path.join(os.path.dirname(script_path), '../../')
normalized_path = os.path.normpath(up_two_levels)
os.chdir(normalized_path) # Change the current working directory to the normalized path

from conf import *
from datetime import datetime
import tensorflow as tf
import pickle


print(os.getcwd())
training.check_tensorflow_gpu()
training.check_tensorflow_version()
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
current_date = datetime.now().strftime("%Y%m%d_%H%M")

discribtion = "2024-08-15 dataset, with bottle neck of 2x2 feature maps"
DATASET = "2024-08-15"
dev_flag = False


if dev_flag:
    ABS_DIR = f"C:/Users/qiyuanxu/Documents/DataHub/datasets/{DATASET}/"
    SAVE_TO = f'C:/Users/qiyuanxu/Documents/DataHub/result/dev/{DATASET}_{current_date}/' 
else:
    ABS_DIR = f'../dataset/{DATASET}/'
    SAVE_TO = f'../results/{DATASET}_{current_date}/' 

DATABASE_ROOT = ABS_DIR + "db/dataset_meta.db"    
log_save_path=SAVE_TO + "logs/"

utils.check_and_create_folder(SAVE_TO)
utils.check_and_create_folder(SAVE_TO+'models')
utils.check_and_create_folder(log_save_path)

# ============================
# Unzip dataset
# ============================
if not dev_flag:
    utils.extract_tar_file(ABS_DIR[:-1]+".tar", "/".join(ABS_DIR.split("/")[:-2]))


# ============================
# Model Construction
# ============================

@tf.keras.utils.register_keras_serializable()
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, 
                             kernel_size=size, 
                             strides=2, 
                             padding='same',
                             kernel_initializer=initializer, 
                             use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

@tf.keras.utils.register_keras_serializable()
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, 
                                    kernel_size=size, 
                                    strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

 

def Autoencoder(input_shape): 
    inputs = tf.keras.layers.Input(shape=input_shape)  # (input image: batch_size, 256, 256, 1)
    encoder = [ 
    downsample(64, 4, apply_batchnorm=False),  # output batch_size, 128, 128, 64
    downsample(128, 4),  # output batch_size, 64, 64, 128
    downsample(256, 4),  # output batch_size, 32, 32, 256 
    downsample(512, 4),  # output batch_size, 16, 16, 512
    downsample(512, 4),  # output batch_size, 8, 8, 512
    downsample(1024, 4),  # output batch_size, 4, 4, 1024
    downsample(1024, 4),  # output batch_size, 2, 2, 1024
    downsample(1024, 4),  # output batch_size, 1, 1, 1024
    ] 

    decoder = [ 
    upsample(1024, 4, apply_dropout=True),  # output batch_size, 2, 2, 1024
    upsample(1024, 4, apply_dropout=True),  # output batch_size, 4, 4, 1024
    upsample(512, 4, apply_dropout=True),  # output batch_size, 8, 8, 512
    upsample(512, 4, apply_dropout=True),  # output batch_size, 16, 16, 512
    upsample(256, 4, apply_dropout=True),  # output batch_size, 32, 32, 256
    upsample(128, 4),  # output batch_size, 64, 64, 128
    upsample(64, 4),  # output batch_size, 128, 128, 64
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
     # output batch_size, 256, 256, 1
    last = tf.keras.layers.Conv2DTranspose(input_shape[-1], kernel_size=4, strides=2, 
                                           kernel_initializer=initializer, activation='sigmoid', padding='same')   # activation='tanh'   activation='sigmoid'
    
    # initializer = tf.random_normal_initializer(0., 0.02)
    # last = tf.keras.layers.Conv2DTranspose(input_shape[-1], 
    #                                        kernel_size =4,
    #                                        strides=2,
    #                                        padding='same',
    #                                        kernel_initializer=initializer,
    #                                        activation='tanh')  # (batch_size, 256, 256, 1)
    
    # without skip connections
    x = inputs 
    for down in encoder: 
        x = down(x) 
    for up in decoder: 
        x = up(x) 
    x = last(x) 
    return tf.keras.Model(inputs=inputs, outputs=x) 


    # with skip connections
    # skips = []
    # x = inputs 
    
    # for down in encoder:
    #     x = down(x)
    #     # print(x.shape)
    #     skips.append(x)
    # skips = reversed(skips[:-1])  # reversed(skips[:-1])
    
    # # x = decoder[0](x) # bottle neck layer direct connection, which is the last output from encoder to the first layer of decoder.
    # for up, skip in zip(decoder, skips):
    #     # print(x.shape, skip.shape)
    #     x = up(x)
    #     x = tf.keras.layers.Concatenate()([x, skip])
    #     # print(x.shape)
        
    # x = last(x)
    # # print(x.shape)
    # return tf.keras.Model(inputs=inputs, outputs=x)
    


# ============================
# Data Preparation
# ============================

batch_size = 4
DB = database.SQLiteDB(DATABASE_ROOT)

# gain_func = utils.preset_kwargs(**{"max_gain":5, "min_gain":1, "scale":10})(apply_intensity_gain)
# steps = []
# load_and_process_image = utils.preset_kwargs(processing_steps=steps)(datapipeline.load_and_process_image)

# creating training set
# sql = """
#     SELECT 
#         id, batch, purpose, image_path, comments
#     FROM 
#         mmf_dataset_metadata
#     WHERE 
#         is_calibration = 0 AND purpose = 'training' AND comments != 'temporal_shift_check'
#     LIMIT 25000
# """

sql = """
    SELECT 
        id, batch, image_path, purpose, comments
    FROM 
        mmf_dataset_metadata
    WHERE 
        is_calibration = 0 and purpose = 'training'
"""

df = DB.sql_select(sql)
print('Total number of records in the table: ' + str(len(df)))
train_paths = [ABS_DIR+i for i in df["image_path"].to_list()]
train_dataset = datapipeline.tf_dataset_prep(train_paths, datapipeline.load_and_process_image, batch_size)
datapipeline.datapipeline_conclusion(train_dataset, batch_size)

# creating validation set
sql = """
    SELECT
        id, batch, purpose, image_path, comments
    FROM 
        mmf_dataset_metadata
    WHERE 
        is_calibration = 0 AND purpose = 'testing' AND comments IS NULL
"""
df = DB.sql_select(sql)
print('Total number of records in the table: ' + str(len(df)))
val_paths = [ABS_DIR+i for i in df["image_path"].to_list()]
val_paths = [val_paths[i] for i in range(0, len(val_paths), 5)]  # (take 20% of the data for validation, the rest will be used for testing, code for testing should corespond to this)
val_dataset = datapipeline.tf_dataset_prep(val_paths, datapipeline.load_and_process_image, batch_size, shuffle=False)
datapipeline.datapipeline_conclusion(val_dataset, batch_size)



# ============================
# Model Training
# ============================

for left_imgs, right_imgs in train_dataset.take(1):
    shape = left_imgs.shape[1:]

autoencoder = Autoencoder(shape)
autoencoder.summary()
print(f"model size: {autoencoder.count_params() * 4 / (1024**2)} MB") 

# Initialize early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8,
                                                  verbose=1, mode='min', restore_best_weights=True)
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)   # successful one: 0.0001
autoencoder.compile(optimizer=adam_optimizer, 
                    loss=tf.keras.losses.MeanSquaredError())

history = autoencoder.fit(  
    train_dataset,  # Dataset already includes batching and shuffling
    epochs=80,
    validation_data=val_dataset,
    callbacks=[training.ImageReconstructionCallback(val_dataset, log_save_path, cmap="viridis"), early_stopping],  # gray, viridis
    verbose=1 if dev_flag else 2  # Less verbose output suitable for large logs
)



# ============================
# Save Model/Results
# ============================

autoencoder.save(SAVE_TO+'models/model.keras')
print('model saved!')
# Save the training history
with open(log_save_path+'training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
# Save all the other information
Logger = training.Logger(log_dir=log_save_path, model=autoencoder, dataset=DATASET, history=history, info=discribtion)
Logger.save()


# ============================
# Delete dataset
# ============================
DB.close()  

utils.delete_path(ABS_DIR[:-1])


