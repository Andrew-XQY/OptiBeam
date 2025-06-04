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
import random


print(os.getcwd())
training.check_tensorflow_gpu()
training.check_tensorflow_version()
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
current_date = datetime.now().strftime("%Y%m%d_%H%M")

discribtion = "CLEAR dataset for both training and evaluation, with best autoencoder model, round 1"
DATASET = "2024-08-15"
dev_flag = True
which_dataset = "1"  # synthetic = 0 or CLEAR = 1, for training and validation set.
CLEAR_data_dir = "C:/Users/qiyuanxu/Documents/DataHub/local_images/MMF"



# ABS_DIR = f"C:/Users/xqiyuan/cernbox/Documents/DataHub/datasets/{DATASET}/"
# SAVE_TO = f'C:/Users/xqiyuan/cernbox/Documents/DataHub/results/dev/{DATASET}_{current_date}/' 
if dev_flag:
    ABS_DIR = f"C:/Users/qiyuanxu/Documents/DataHub/datasets/{DATASET}/"
    SAVE_TO = f'C:/Users/qiyuanxu/Documents/DataHub/results/dev/{DATASET}_{current_date}/' 
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
    downsample(1024, 4),  # output batch_size, 8, 8, 1024
    downsample(1024, 4),  # output batch_size, 4, 4, 1024
    ] 

    decoder = [ 
    upsample(1024, 4, apply_dropout=True),  # output batch_size, 8, 8, 1024
    upsample(1024, 4, apply_dropout=True),  # output batch_size, 16, 16, 1024
    upsample(512, 4, apply_dropout=True),  # output batch_size, 32, 32, 512
    upsample(256, 4),  # output batch_size, 64, 64, 256
    upsample(128, 4),  # output batch_size, 128, 128, 128
    upsample(64, 4),  # output batch_size, 256, -)  extra layer, actually can be merged with the 'Conv2D' below
    ]
    
    last = tf.keras.layers.Conv2D(input_shape[-1], kernel_size=4, activation='sigmoid', padding='same')   # activation='tanh'   activation='sigmoid'
    
    # without skip connections
    x = inputs 
    for down in encoder: 
        x = down(x) 
    for up in decoder: 
        x = up(x) 
    x = last(x) 
    return tf.keras.Model(inputs=inputs, outputs=x) 


# ============================
# Data Preparation 
# ============================
batch_size = 4

if which_dataset == "0": # synthetic dataset
    print("Using synthetic dataset for training.")
    print(DATABASE_ROOT)
    DB = database.SQLiteDB(DATABASE_ROOT)
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

elif which_dataset == "1":  # CLEAR dataset
    print("Using CLEAR dataset for training.")
    CLEAR_data = utils.get_all_file_paths(CLEAR_data_dir)
    random.seed(42)
    random.shuffle(CLEAR_data)
    n = len(CLEAR_data)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_paths = CLEAR_data[:n_train]
    val_paths   = CLEAR_data[n_train:n_train + n_val]
    # test  = CLEAR_data[n_train + n_val:]
    train_dataset = datapipeline.tf_dataset_prep(train_paths, datapipeline.load_and_process_image, batch_size)
    datapipeline.datapipeline_conclusion(train_dataset, batch_size)
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


