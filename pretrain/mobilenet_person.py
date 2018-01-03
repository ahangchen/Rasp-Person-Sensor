from random import shuffle

import keras
from keras import Input
import numpy as np
import os
import cv2
from keras.applications.mobilenet import preprocess_input, relu6, DepthwiseConv2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.engine import Model
from keras.layers import Dropout, Dense
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

import util.cuda_util


def mobilenet_binary():
    model = keras.applications.mobilenet.MobileNet(include_top=True, weights='imagenet',
                                                   input_tensor=Input(shape=(224, 224, 3)))
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.output_layers = [model.layers[-1]]
    x = model.output
    x = Dense(2, activation='softmax', name='fc8')(x)
    net = Model(inputs=[model.input], outputs=[x])
    return net


def part_img(img, rand_idx):
    height = img.shape[0]
    if rand_idx % 3 == 2:
        return cv2.resize(img[height / 5 * 2: height / 5 * 3, :, :], (224, 224))
    elif rand_idx % 3 == 1:
        return cv2.resize(img[height / 4 * 2: height / 4 * 3, :, :], (224, 224))
    elif rand_idx % 3 == 0:
        return cv2.resize(img[height / 4 * 3:, :, :], (224, 224))


def load_data(train_dir):
    images, labels = [], []
    for i, image_name in enumerate(sorted(os.listdir(train_dir))):
        img = image.load_img(os.path.join(train_dir, image_name), target_size=[224, 224])
        img = image.img_to_array(img)
        rand_seed = i % 2
        if rand_seed == 0:
            img = part_img(img, i)
            if i % 500 == 0:
                cv2.imwrite(image_name, img)
            labels.append(0)
        else:
            labels.append(1)
        if img is None:
            print image_name
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        images.append(img[0])

    img_cnt = len(labels)
    shuffle_idxes = range(img_cnt)
    shuffle(shuffle_idxes)
    shuffle_imgs = list()
    shuffle_labels = list()
    for idx in shuffle_idxes:
        shuffle_imgs.append(images[idx])
        shuffle_labels.append(labels[idx])
    images = np.array(shuffle_imgs)
    labels = to_categorical(shuffle_labels)
    return images, labels


def train(net, batch_size=64):
    train_datagen = ImageDataGenerator(
        rotation_range=0.2)
    val_datagen = ImageDataGenerator()
    images, labels = load_data('/home/cwh/coding/Market-1501/train')
    img_cnt = len(images)
    print img_cnt
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    auto_lr = ReduceLROnPlateau(monitor='val_loss', patience=3)
    net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    net.fit_generator(
        train_datagen.flow(images[: int(0.9 * img_cnt)], labels[: int(0.9 * img_cnt)], batch_size=batch_size),
        steps_per_epoch=len(images) / batch_size + 1, epochs=5,
        validation_data=val_datagen.flow(images[int(0.9 * img_cnt):], labels[int(0.9 * img_cnt):],
                                         batch_size=batch_size),
        validation_steps=img_cnt / 10 / batch_size + 1, callbacks=[early_stopping, auto_lr])
    net.save('mbp.h5')
    return net


def single_img_eval(net):
    dir_path = '/home/cwh/coding/Market-1501/test'
    for i, image_name in enumerate(sorted(os.listdir(dir_path))):
        img = image.load_img(os.path.join(dir_path, image_name), target_size=[224, 224])
        img = image.img_to_array(img)
        rand_seed = i % 3
        if rand_seed == 2 or rand_seed == 1:
            img = part_img(img, rand_seed)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        y = net.predict(img)
        print '%d %s' % (np.argmax(y), image_name)
        # print '%d %s' % (y > 0.5, image_name)


if __name__ == '__main__':
    # net = mobilenet_binary()
    # net = train(net)
    net = load_model('mbp_sig.h5', custom_objects={'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D})
    single_img_eval(net)
