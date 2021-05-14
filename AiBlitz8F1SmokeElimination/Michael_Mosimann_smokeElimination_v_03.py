# Code from https://www.tensorflow.org/tutorials/generative/pix2pix .
# Full credit to pix2pix.
import tensorflow as tf
import os
import time
import datetime
from matplotlib import pyplot as plt
from IPython import display


def load(image_file):
    # Load
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    # real_image = tf.cast(real_image, tf.float32)
    return image


def load_2(image_file_smoke, image_file_clear):
    # Load
    # print(image_file_smoke)
    # print("test")
    # image_file_clear = image_file_smoke.replace("smoke", "clear")
    print(image_file_smoke)
    print(image_file_clear)
    image_smoke = tf.io.read_file(image_file_smoke)
    image_smoke = tf.image.decode_jpeg(image_smoke)
    image_smoke = tf.cast(image_smoke, tf.float32)
    image_clear = tf.io.read_file(image_file_clear)
    image_clear = tf.image.decode_jpeg(image_clear)
    image_clear = tf.cast(image_clear, tf.float32)
    # real_image = tf.cast(real_image, tf.float32)
    return image_smoke, image_clear


# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


# read data, datasets

def createDataSet(path_name, batchSize):
    dataset = tf.data.Dataset.list_files(path_name+'/*.jpg', shuffle=False)
    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batchSize)
    return dataset


def createDataSet_2(path_name_smoke, path_name_clear, batchSize):
    # dataset = tf.data.Dataset.list_files((path_name_smoke+'/*.jpg', path_name_clear+'/*.jpg'), shuffle=False)
    dataset = tf.data.Dataset.list_files(path_name_smoke+'/*.jpg', shuffle=False)
    dataset_2 = tf.data.Dataset.list_files(path_name_clear+'/*.jpg', shuffle=False)
    # dataset = dataset.list_files(path_name_clear+'/*.jpg', shuffle=False)
    # for i in dataset:
    #    print(i)
    # print(len(os.listdir(path_name_smoke)))
    # file_names_smoke = os.listdir(path_name_smoke)
    # file_names_clear = os.listdir(path_name_clear)
    # dataset_clear = tf.data.Dataset.list_files(path_name_clear+'/*.jpg', shuffle=False)
    # dataset = dataset.map(load_2, num_parallel_calls=tf.data.AUTOTUNE)
    print(dataset)
    print(dataset_2)
    dataset = tf.data.Dataset.map(load_2(dataset, dataset_2),
                                  num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batchSize)
    return dataset


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


def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
        ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
        ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(output_channel, 4, strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)
    x = inputs
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output),
                           disc_generated_output)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)
    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output),
                                 disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def generate_images_without_target(model, test_input):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))
    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     discriminator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))
        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()
        display.clear_output(wait=True)
        for example_input in test_ds.take(1):
            generate_images_without_target(generator, example_input)
        print("Epoch: ", epoch)
        # Train
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            train_step(input_image, target, epoch)
        print()
        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time()-start))
    checkpoint.save(file_prefix=checkpoint_prefix)


buffer_size = 400
batch_size = 1
image_width, image_height = 256, 256
output_channel = 3  # Variable globale
LAMBDA = 100  # Variable globale
EPOCHS = 150  # Variable globale
# File paths
data_directiory = "data"
train_clear_path = os.path.join(data_directiory, "train/clear")
train_smoke_path = os.path.join(data_directiory, "train/smoke")
val_clear_path = os.path.join(data_directiory, "val/clear")
val_smoke_path = os.path.join(data_directiory, "val/smoke")
test_data_path = os.path.join(data_directiory, "test/smoke")
test_submission_path = "clear"
# Single image to test code parts.
smoke_img = load(os.path.join(train_clear_path, "0.jpg"))
# Datasets
train_dataset_clear = createDataSet(train_clear_path, batch_size)
train_dataset_smoke = createDataSet(train_smoke_path, batch_size)
train_dataset = createDataSet_2(train_smoke_path, train_clear_path, batch_size)
val_dataset_clear = createDataSet(val_clear_path, batch_size)
val_dataset_smoke = createDataSet(val_smoke_path, batch_size)
val_dataset = createDataSet_2(val_smoke_path, val_clear_path, batch_size)
test_dataset_smoke = createDataSet(test_data_path, batch_size)
# Downsampling
down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(smoke_img, 0))
print(down_result.shape)
up_model = upsample(3, 4)
up_result = up_model(down_result)
print(up_result.shape)
# Generator
generator = Generator()
gen_output = generator(smoke_img[tf.newaxis, ...], training=False)
plt.imshow(gen_output[0, ...])  # Show loss of generator
# plt.pause(2)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# Discriminator
discriminator = Discriminator()
disc_out = discriminator([smoke_img[tf.newaxis, ...], gen_output], training=False)
# plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
# plt.colorbar()
# plt.pause(2)
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
# for example_input, example_target in train_dataset_smoke.take(1):
for example_input, example_target in train_dataset.take(1):
    generate_images(generator, example_input, example_target)
log_dir="logs/"
summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# smoke_img = load(os.path.join(train_clear_path, "0.jpg"))
# clear_img = load(os.path.join(train_clear_path, "0.jpg"))
# casting to int for matplotlib to show the image
fit(train_dataset, EPOCHS, test_dataset_smoke)
