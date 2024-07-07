import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

CHANNELS = 1 # the number of channels in the input image
LAMBDA = 100 # the weight of the pixel level mean absolute error in the generator loss function

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

# __________________________________________________________
# build the upsampler (decoder part)
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result



'''
define the Generator structure, plotting the model graph
'''

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, CHANNELS])
    down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(128, 4),  # (batch_size, 32, 32, 256)
    downsample(256, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    ]
    up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)
    x = inputs # x is the temp variable to go through the list.
    # Downsampling through the model
    skips = []
    for down in down_stack:  # connnected in forward sequence
        # if the first layer is the input x, then the ouput of second layer can be considered as f(x)
        # similarly, the thrid layer is g(f(x)), and this will corresponde as the input of next layer.
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
    #x = tf.keras.layers.Concatenate()([x, skip])  # only skip connect one, or no skip connection at all
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


'''
define the Discriminator structure (PatchGAN classifier), plotting the model graph
'''

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, CHANNELS], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, CHANNELS], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)  concatenate these 2 inputs together.

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


'''Define the Generator Loss function for training'''

def generator_loss(disc_generated_output, gen_output, target):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

# the loss of the generator contains two parts, the first one is the Pixel level mean absolute error, it is scaled and 
# combined with the discriminator loss using a lambda as the total loss for updating the generator.


'''Define the loss function for the Discriminator'''

# notic, disc_real_output is the output of the discriminator, in this case, is a 30x30 map.
def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # this will output a single value represent the average over the entire map
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output) 
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


# during training, demo result images
# def generate_images(model, test_input, tar):
#     prediction = model(test_input, training=True) # when on testset, this still should be True
#     plt.figure(figsize=(15, 15))

#     display_list = [test_input[0], tar[0], prediction[0]]
#     title = ['MMF Speckle Pattern (Input)', 'Original Beam Distribution (Ground Truth)',
#              'Reconstructed Image (Output)']

#     for i in range(3):  # present the result in a nice visual way
#         plt.subplot(1, 3, i+1)
#         plt.title(title[i])
#         # Getting the pixel values in the [0, 1] range to plot. 
#         # plt.imshow(display_list[i] * 0.5 + 0.5)
#         plt.imshow(display_list[i] * 0.5 + 0.5, cmap='Greys_r') # original image
#         plt.axis('off')
#     plt.show()


# def generate_images(model, test_input, tar, save_path):
#     prediction = model(test_input, training=True)  # Keep training=True even on the test set

#     display_list = [test_input[0], tar[0], prediction[0]]
#     title = ['MMF Speckle Pattern (Input)', 'Original Beam Distribution (Ground Truth)',
#              'Reconstructed Image (Output)']

#     # Get the current timestamp to use in file names
#     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

#     # Save each of the images in the display list to the specified path
#     for i in range(3):
#         plt.figure(figsize=(5, 5))  # Create a new figure for each image
#         plt.title(title[i])
#         plt.imshow(display_list[i] * 0.5 + 0.5, cmap='Greys_r')
#         plt.axis('off')
#         # Build the file path using the timestamp and index to avoid overwriting
#     file_path = f"{save_path}/{timestamp}.png"
#     plt.savefig(file_path)  # Save the figure to the file path
#     plt.close()  # Close the plot to free up memory
#     plt.clf()



def generate_images(model, test_input, tar, save_path):
    prediction = model(test_input, training=True)  # Keep training=True even on the test set

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['MMF Speckle Pattern (Input)', 'Original Beam Distribution (Ground Truth)', 
             'Reconstructed Image (Output)']

    # Create one figure with three subplots
    plt.figure(figsize=(15, 5))  # Adjust the overall size as needed

    # Plot each image in a subplot
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5, cmap='Greys_r')
        plt.axis('off')

    # Get the current timestamp to use in the file name
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = f"{save_path}/{timestamp}.png"
    plt.savefig(file_path)  # Save the figure with all subplots
    plt.close()  # Close the plot to free up memory



