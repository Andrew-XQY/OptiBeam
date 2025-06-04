
# before 2025-0306 the best model structure.
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









# new balanced model (with the same layer number in both encoder and decoder, both 6 layers)

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
    

























