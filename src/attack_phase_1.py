## attack_phase_1.py: generate adversarial examples on the undefended model

## Run this script first, to generate adversarial examples that fool the undefended
## model. We save every potential adversarial example, because we're going for a
## brain-dead brute-force attack. This way we don't have to even know how the
## defended model works and can still successfully attack it.

import sys
import cv2
import numpy as np
import os
import re
import tensorflow as tf
import time

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Activation, Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model, Model
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils.data_utils import get_file

def vgg_model(restore):
    img_input = Input(shape=(224, 224, 3), batch_shape=(1, 224, 224, 3))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, name='fc6')(x)
    x = Activation('relu', name='fc6/relu')(x)
    x = Dense(4096, name='fc7')(x)
    x = Activation('relu', name='fc7/relu')(x)
    x = Dense(2622, name='fc8')(x)

    model = Model(img_input, x)
    model.load_weights(restore)

    return model

class VGGFaceModel:
    def __init__(self, restore, defense=False, session=None):
        self.num_channels = 3
        self.image_size = 224
        self.num_labels = 2622
        self.model = vgg_model(restore)

    def predict(self, img):
        scaled = (img + 0.5) * 255
        return self.model(scaled)


def generate_data(img_idx):
    inputs = []

    img_name = sorted(os.listdir("../data/images/"))[img_idx]
    print("Load", img_name)
    img_path = "../data/images/" + img_name

    img = image.load_img(img_path)
    x = image.img_to_array(img)

    inputs.append((x / 255) - .5)

    inputs = np.array(inputs)

    return inputs


loss_fn = sys.argv[1]
bound = float(sys.argv[2])
step_size = float(sys.argv[3])
rand = int(sys.argv[4])
    
with tf.Session() as sess:
    weights_path = '../data/rcmalli_vggface_tf_vgg16.h5'
    vgg_model = VGGFaceModel(weights_path, defense=False, session=sess)

    xs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    ys = tf.placeholder(tf.float32, [None, None])
    preds = vgg_model.predict(xs)

    # try two different loss functions
    if loss_fn == 'xent':
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=ys)
    else:
        loss = -tf.reduce_sum(preds*ys,axis=1)
        loss += tf.reduce_max(preds*(1-ys) - ys*10000)
    grads = tf.sign(tf.gradients(loss, [xs])[0])

    off = 0
    BS = 10

    # try each potential target class for good measure.
    # usually we don't need very many successes but let's just do them all.
    for target in range(2622):
        inputs = np.array([generate_data(i+off)[0] for i in range(BS)])
        targets = np.zeros((BS, 2622))
        targets[:, target] = 1
        
        orig = np.copy(inputs)

        print(np.argmax(sess.run(preds, {xs: orig})))
        # 100 iterations of PGD
        for step in range(100):
            g = sess.run(grads, {xs: inputs +np.random.normal(0,.005,size=inputs.shape)*rand,
                                 ys: targets})
            inputs -= g*0.1/255*bound*step_size
            inputs = np.clip(inputs, orig-1./255*bound, orig+1./255*bound)
            if step%10 == 0:
                p = sess.run(preds, {xs: inputs})
                pp = np.copy(p)
                pp[np.arange(BS),np.argmax(targets,axis=1)] = -10000
                print(np.argmax(p,axis=1), p[np.arange(BS),np.argmax(targets)], np.max(pp,axis=1))

        print()

        p = sess.run(preds, {xs: inputs})

        # did it work against the undefended model?
        for img in range(BS):
            if np.argmax(p[img])==target:
                print("Success", "tmp/%s%f_adv_%d_target_%d.npy"%(loss_fn,bound,img,target))
                np.save("tmp/%s%f_adv_%d_target_%d_s%0.2f.npy"%(loss_fn,bound,img,target, step_size), inputs[img]-orig[img])
