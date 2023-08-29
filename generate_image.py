
import tensorflow as tf
from tensorflow.keras import layers, models

import time
import matplotlib.pyplot as plt
import numpy as np

def Discriminator(input_shape=[28,28,1], n_classes=10, filters=32):
    # input label (Embedding layers)
    input_label = layers.Input(shape=[1])
    em_block = layers.Embedding(n_classes, 64)(input_label)
    em_block = layers.Dense(input_shape[0]*input_shape[1])(em_block)
    em_block = layers.Reshape(input_shape)(em_block)
    
    # convolutional layers
    input_img = layers.Input(shape=input_shape)
    ## merge label and image
    merge = layers.Concatenate()([input_img, em_block])
    
    ## convolutional block 1
    conv_1 = layers.Conv2D(filters, 3, padding='same')(merge)
    conv_1 = layers.BatchNormalization()(conv_1)
    conv_1 = layers.ReLU()(conv_1)
    conv_1 = layers.Dropout(0.3)(conv_1)
    conv_1 = layers.MaxPooling2D()(conv_1)
    
    ## convolutional block 2
    conv_2 = layers.Conv2D(filters * 2, 3, padding='same')(conv_1)
    conv_2 = layers.BatchNormalization()(conv_2)
    conv_2 = layers.ReLU()(conv_2)
    conv_2 = layers.Dropout(0.3)(conv_2)
    conv_2 = layers.MaxPooling2D()(conv_2)
    
    ## convolutional block 3
    conv_3 = layers.Conv2D(filters * 4, 3, padding='same')(conv_2)
    conv_3 = layers.BatchNormalization()(conv_3)
    conv_3 = layers.ReLU()(conv_3)
    conv_3 = layers.Dropout(0.3)(conv_3)
    conv_3 = layers.MaxPooling2D()(conv_3)
    
    flat = layers.Flatten()(conv_3)
    
    out = layers.Dense(128)(flat)
    out = layers.ReLU()(out)
    out = layers.Dropout(0.3)(out)
    out = layers.Dense(1)(out)
    
    return models.Model([input_label, input_img], out)

discriminator = Discriminator()
# tf.keras.utils.plot_model(discriminator, show_shapes=True, show_trainable=True)

LATENT_DIM = 256

def Generator(latent_dim, n_classes=10):
    # input label (Embedding layers)
    input_label = layers.Input(shape=[1])
    em_block = layers.Embedding(n_classes, 64)(input_label)
    em_block = layers.Dense(7*7)(em_block)
    em_block = layers.Reshape([7, 7, 1])(em_block)
    
    # input noise
    input_latent = layers.Input(shape=[latent_dim])
    
    full_c = layers.Dense(7*7*latent_dim)(input_latent)
    full_c = layers.LeakyReLU()(full_c)
    full_c = layers.Reshape([7, 7, latent_dim])(full_c)
    
    # merge label and latent
    merge = layers.Concatenate()([em_block, full_c])
    
    # convolutional transpose block 1
    conv_trans_1 = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False)(merge)
    conv_trans_1 = layers.BatchNormalization()(conv_trans_1)
    conv_trans_1 = layers.LeakyReLU()(conv_trans_1)
    
    # convolutional transpose block 2
    conv_trans_2 = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False)(conv_trans_1)
    conv_trans_2 = layers.BatchNormalization()(conv_trans_2)
    conv_trans_2 = layers.LeakyReLU()(conv_trans_2)
    
    # output
    out = layers.Conv2D(1, (5,5), activation='tanh', padding='same')(conv_trans_2)
    
    return models.Model([input_label, input_latent], out)
    

generator = Generator(LATENT_DIM)
# tf.keras.utils.plot_model(generator, show_shapes=True, show_trainable=True)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_prediction):
    return cross_entropy(tf.ones_like(fake_prediction), fake_prediction)

def discriminator_loss(real_prediction, fake_prediction):
    real_loss = cross_entropy(tf.ones_like(real_prediction), real_prediction)
    fake_loss = cross_entropy(tf.zeros_like(fake_prediction), fake_prediction)
    total_loss = real_loss + fake_loss
    return total_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

import os
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore('training_checkpoints/ckpt-1').expect_partial()

def generate_images(model, epoch, test_input, test_label):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model([test_label, test_input], training=False)

    input_length = test_input.shape[0]
    fig = plt.figure(figsize=(input_length, 1))

    for i in range(predictions.shape[0]):
        plt.subplot(1, input_length, i+1)
        pred = 255.0-(predictions[i, :, :, :] * 127.5 + 127.5)
        plt.imshow(pred, cmap='gray')
        plt.axis('off')
        plt.tight_layout(pad=0)

    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

u_input = 0
while True:
    u_input = input('enter your number: ')
    if u_input.isnumeric():
        u_input = list(u_input)
        u_input = list(map(int, u_input))
        break

label = np.array(u_input)
print(label)
seed = tf.random.normal([label.size, LATENT_DIM])
generate_images(generator, 0, seed, label)