import PIL
import glob

import numpy as np
import tensorflow as tf

from datetime import datetime
from tensorflow.keras import Sequential, layers
from tf_gen_models.algorithms.gan import GAN
from tf_gen_models.callbacks import GanExpLrScheduler, ImageSaver, ModelSaver


# +---------------------------------+
# |    Load and prepare datasets    |
# +---------------------------------+

(train_img, _), (test_img, _) = tf.keras.datasets.mnist.load_data()

train_img = train_img . reshape ( train_img.shape[0], 28, 28, 1 ) \
                      . astype ( np.float32 )
train_img = (train_img - 127.5) / 127.5   # pixel intensity in [-1,1]

test_img = test_img . reshape ( test_img.shape[0], 28, 28, 1 ) \
                    . astype ( np.float32 )
test_img = (test_img - 127.5) / 127.5   # pixel intensity in [-1,1]

BUFFER_SIZE = 60000
BATCH_SIZE = 64

## TF.DATA.DATASET

train_ds = ( 
  tf.data.Dataset.from_tensor_slices ( train_img )
  .shuffle ( BUFFER_SIZE )                       # shuffle all the images
  .batch ( BATCH_SIZE, drop_remainder = True )   # mini-batch splitting
  .cache()                                       # cache the dataset
  # .prefetch ( tf.data.AUTOTUNE )   # pre-prepare data to be consumed
)

test_ds = ( 
  tf.data.Dataset.from_tensor_slices ( test_img )
  .shuffle ( BUFFER_SIZE )                       # shuffle all the images
  .batch ( BATCH_SIZE, drop_remainder = True )   # mini-batch splitting
  .cache()                                       # cache the dataset
  # .prefetch ( tf.data.AUTOTUNE )   # pre-prepare data to be consumed
)

# +---------------------------+
# |    Adversarial players    |
# +---------------------------+

LATENT_DIM = 100

## GENERATOR

generator = Sequential ( name = "generator" )

generator . add ( layers.Dense ( 7 * 7 * 256, use_bias = False, input_shape = (LATENT_DIM,) ) )
generator . add ( layers.BatchNormalization() )
generator . add ( layers.LeakyReLU() )

generator . add ( layers.Reshape ( (7, 7, 256) ) )

generator . add ( layers.Conv2DTranspose ( 256, (3, 3), strides = (1, 1), padding = "valid" ) )
generator . add ( layers.BatchNormalization ( axis = 1 ) )
generator . add ( layers.LeakyReLU() )

generator . add ( layers.Conv2DTranspose ( 128, (4, 4), strides = (2, 2), padding = "valid" ) )
generator . add ( layers.BatchNormalization ( axis = 1 ) )
generator . add ( layers.LeakyReLU() )

generator . add ( layers.Conv2DTranspose ( 64, (5, 5), strides = (1, 1), padding = "valid" ) )
generator . add ( layers.BatchNormalization ( axis = 1 ) )
generator . add ( layers.LeakyReLU() )

generator . add ( layers.Conv2DTranspose ( 1, (5, 5), strides = (1, 1), padding = "valid", activation = "tanh" ) )

## DISCRIMINATOR

discriminator = Sequential ( name = "discriminator" )
    
discriminator . add ( layers.Conv2D ( 32, (4, 4), strides = (2, 2), padding = "same", input_shape = [28, 28, 1] ) )
discriminator . add ( layers.BatchNormalization ( axis = 1 ) )
discriminator . add ( layers.LeakyReLU ( alpha = 0.2 ) )

discriminator . add ( layers.Conv2D ( 64, (4, 4), strides = (2, 2), padding = "same" ) )
discriminator . add ( layers.BatchNormalization ( axis = 1 ) )
discriminator . add ( layers.LeakyReLU ( alpha = 0.2 ) )

discriminator . add ( layers.Conv2D ( 128, (4, 4), strides = (2, 2), padding = "same" ) )
discriminator . add ( layers.BatchNormalization ( axis = 1 ) )
discriminator . add ( layers.LeakyReLU ( alpha = 0.2 ) )

discriminator . add ( layers.Flatten() )
discriminator . add ( layers.Dense ( 1, activation = "sigmoid" ) )

# +--------------------------+
# |    Training procedure    |
# +--------------------------+

gan = GAN (generator, discriminator, latent_dim = LATENT_DIM)

gan . summary()

## OPTIMIZERS

g_opt = tf.keras.optimizers.Adam ( 1e-5 )
d_opt = tf.keras.optimizers.Adam ( 1e-5 )

gan . compile ( g_optimizer = g_opt , 
                d_optimizer = d_opt ,
                g_updt_per_batch = 1 ,
                d_updt_per_batch = 3 )

## CALLBACKS

lr_sched  = GanExpLrScheduler ( factor = 0.90, step = 5 )
img_saver = ImageSaver ( name = "dc-gan", dirname = "./images/dc-gan", step = 1, look = "multi" )
# mod_saver = ModelSaver ( name = "dc-gan", dirname = "./models", step = 10 )

## TRAINING

EPOCHS = 50
STEPS_PER_EPOCH = int ( len(train_img) / BATCH_SIZE )

start = datetime.now()

train = gan . fit ( train_ds , 
                    epochs = EPOCHS ,
                    steps_per_epoch = STEPS_PER_EPOCH ,
                    validation_data = test_ds ,
                    callbacks = [ lr_sched, img_saver ] ,
                    verbose = 1 )

stop = datetime.now()

timestamp = str(stop-start) . split (".") [0]   # HH:MM:SS
timestamp = timestamp . split (":")   # [HH, MM, SS]
timestamp = f"{timestamp[0]}h {timestamp[1]}min {timestamp[2]}s"

print (f"Model training completed in {timestamp}.")

# +--------------------+
# |    Create a GIF    |
# +--------------------+

anim_file = "./images/dc-gan.gif"

filenames = glob.glob ("./images/dc-gan/dc-gan_ep*.png")
filenames = sorted (filenames)

img , *imgs = [ PIL.Image.open(f) for f in filenames ]
img . save ( fp = anim_file, format = "GIF", append_images = imgs,
             save_all = True, duration = 135, loop = 0 )

print (f"GIF correctly exported to {anim_file}.")
