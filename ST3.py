from __future__ import print_function, division

from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AvgPool2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from skimage.transform import resize

import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from ST1 import VGG16_AvgPool, VGG16_AvgPool_CutOff, unpreprocess, scale_img
from ST2 import gram_matrix, style_loss, minimize

from scipy.optimize import fmin_l_bfgs_b


def load_img_and_preprocess(path, shape=None):
    img = image.load_img(path, target_size=shape)

    # convert image to array and preprocess for vgg
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


content_img = load_img_and_preprocess(
    r'C:\Users\aayus\Desktop\NeuralStyleTransfer\parrot.jpg',
)

# resize the style image
# since we don't care too much about warping it
h, w = content_img.shape[1:3]
style_img = load_img_and_preprocess(
    r'C:\Users\aayus\Desktop\NeuralStyleTransfer\starrynight.jpg',
    (h, w)
)

# we'll use this throughout the rest of the script
batch_shape = content_img.shape
shape = content_img.shape[1:]

# we want to make only 1 VGG here
# as you'll see later, the final model needs
# to have a common input
vgg0 = VGG16_AvgPool(shape)

content_model = Model(vgg0.input, vgg0.layers[13].get_output_at(0))
content_target = K.variable(content_model.predict(content_img))

# create the style model
# we want multiple outputs
# we will take the same approach as in style_transfer2.py
symbolic_conv_outputs = [
    layer.get_output_at(1) for layer in vgg0.layers \
    if layer.name.endswith('conv1')
]

# make a big model that outputs multiple layers' outputs
style_model = Model(vgg0.input, symbolic_conv_outputs)

# calculate the targets that are output at each layer
style_layers_outputs = [K.variable(y) for y in style_model.predict(style_img)]

# we will assume the weight of the content loss is 1
# and only weight the style losses
style_weights = [0.2, 0.4, 0.3, 0.5, 0.2]

# create the total loss which is the sum of content + style loss
loss = K.mean(K.square(content_model.output - content_target))

for w, symbolic, actual in zip(style_weights, symbolic_conv_outputs, style_layers_outputs):
    # gram_matrix() expects a (H, W, C) as input
    loss += w * style_loss(symbolic[0], actual[0])

# once again, create the gradients and loss + grads function
# note: it doesn't matter which model's input you use
# they are both pointing to the same keras Input layer in memory
grads = K.gradients(loss, vgg0.input)

# just like theano.function
get_loss_and_grads = K.function(
    inputs=[vgg0.input],
    outputs=[loss] + grads
)


def get_loss_and_grads_wrapper(x_vec):
    l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
    return l.astype(np.float64), g.flatten().astype(np.float64)


final_img = minimize(get_loss_and_grads_wrapper, 30, batch_shape)
plt.imshow(scale_img(final_img))
plt.show()
