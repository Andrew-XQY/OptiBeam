import os
script_path = os.path.abspath(__file__)  # Get the absolute path of the current .py file
up_two_levels = os.path.join(os.path.dirname(script_path), '../../')
normalized_path = os.path.normpath(up_two_levels)
os.chdir(normalized_path) # Change the current working directory to the normalized path

import model
from conf import *
from tqdm import tqdm
import cv2
import tensorflow as tf
import numpy as np 
import datetime, time

print(os.getcwd())
current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
DATASET = "2024-08-15"
# SAVE_TO = '../results/'
# ABS_DIR = f"C:/Users/qiyuanxu/Documents/DataHub/datasets/{DATASET}/"
SAVE_TO = f'C:/Users/xqiyuan/cernbox/Documents/DataHub/results/dev/{DATASET}_{current_date}/' 
IMAGE_SHAPE = (256, 256, 1)

log_save_path=SAVE_TO + "logs/"
utils.check_and_create_folder(SAVE_TO)
utils.check_and_create_folder(SAVE_TO+'models')
utils.check_and_create_folder(log_save_path)



# training.check_tensorflow_gpu() # check if the GPU is available
training.check_tensorflow_gpu()

# ------------------------------ prepare datasets ------------------------------
def read_images_as_grayscale(directory):
    files = [file for file in os.listdir(directory) if file.endswith('.png')]
    images = []
    for filename in tqdm(files, desc='Reading images'):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def image_generator(images):
    for img in images:
        # Normalize the image
        l, r = utils.split_image(img)
        l = l[np.newaxis, ..., np.newaxis]
        r = r[np.newaxis, ..., np.newaxis]
        yield r.astype('float32') / 255.0, l.astype('float32') / 255.0


train_images = read_images_as_grayscale(f'C:/Users/xqiyuan/cernbox/Documents/DataHub/datasets/2024-08-15/dataset/1')
test_images = read_images_as_grayscale(f'C:/Users/xqiyuan/cernbox/Documents/DataHub/datasets/2024-08-15/dataset/2')

# Create datasets using the from_generator method
shape = [1, *IMAGE_SHAPE]
datasets = []
total_sample = len(train_images)

for i in [train_images, test_images]:
    datasets.append(tf.data.Dataset.from_generator(lambda i=i: image_generator(i),
                                                   output_types=(tf.float32, tf.float32),
                                                   output_shapes=(shape, shape)))

train_dataset, test_dataset = datasets

for inp, re in train_dataset.take(1):
    inp, re = inp[0], re[0]
    print("Input shape:", inp.shape)
    print("Reconstructed shape:", re.shape)
    break
# ------------------------------------------------------------------------------




# ------------------------------ prepare models --------------------------------

# testing case
down_model = model.downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print(down_result.shape)
up_model = model.upsample(3, 4)
up_result = up_model(down_result)
print(up_result.shape)

generator = model.Generator()
discriminator = model.Discriminator()
# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64, to_file=SAVE_TO + 'generator.png')
# tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64, to_file=SAVE_TO + 'discriminator.png')

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# for example_input, example_target in train_dataset:
#     model.generate_images(generator, example_input, example_target)
#     break
# ------------------------------------------------------------------------------





# ------------------------------ train models ----------------------------------

log_dir=SAVE_TO + f"logs/"
summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # first get each output from the Generator and discriminator (three in total)
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        # based on the output, calculate the loss respectively
        gen_total_loss, gen_gan_loss, gen_l1_loss = model.generator_loss(disc_generated_output, gen_output, target)
        disc_loss = model.discriminator_loss(disc_real_output, disc_generated_output)
    
    # using SGD propogate the loss back to the network to update the weights
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
    # save the loss history
    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

def fit(train_ds, test_ds, steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()
    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')
            start = time.time()
            model.generate_images(generator, example_input, example_target, save_path=SAVE_TO + "logs/")
            print(f"Step: {step//1000}k")
        train_step(input_image, target, step)
        # Training step
        if (step+1) % 10 == 0:
            print('.', end='', flush=True)
        # Save (checkpoint) the model every 5k steps, disable if it consume too much space
        # if (step + 1) % 5000 == 0:
        #    checkpoint.save(file_prefix=checkpoint_prefix)
    print('training complete!')
    

'''Start training the model'''
# if the total training set is 800, since they recommand batch size of 1 image
# so 40,000 steps (batchs) is equal to 50 epochs (full rounds).
epoch = 80
training_steps = epoch * total_sample
print('total training steps: ', training_steps//1000)
fit(train_dataset, test_dataset, steps=training_steps)
# ------------------------------------------------------------------------------



# ------------------------------ save models -----------------------------------
current_time_seconds = time.time()
timestamp = str(current_time_seconds * 1e9)
model_name = f'{timestamp}.keras'
generator.save(SAVE_TO + f'models/{model_name}')
print('model saved!')


