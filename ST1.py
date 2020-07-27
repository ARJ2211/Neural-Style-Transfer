from __future__ import print_function, division


from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_l_bfgs_b


def VGG16_AvgPool(s):
    # we want to account for features across the entire image
    # so get rid of the max pool which throws away information
    vgg = VGG16(input_shape=s, weights='imagenet', include_top=False)

    new_model = Sequential()
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            # replace it with average pooling
            new_model.add(AveragePooling2D())
        else:
            new_model.add(layer)

    return new_model


def VGG16_AvgPool_CutOff(s, num_convs):
    # there are 13 convolutions in total
    # we can pick any of them as the "output"
    # of our content model

    if num_convs < 1 or num_convs > 13:
        print("num_convs must be in the range [1, 13]")
        return None

    model = VGG16_AvgPool(s)
    new_model = Sequential()
    n = 0
    for layer in model.layers:
        if layer.__class__ == Conv2D:
            n += 1
        new_model.add(layer)
        if n >= num_convs:
            break

    return new_model


def unpreprocess(img0):
    img0[..., 0] += 103.939
    img0[..., 1] += 116.779
    img0[..., 2] += 126.68
    final = img0[..., ::-1]
    return final


def scale_img(xx):
    xx = xx - xx.min()
    xx = xx / xx.max()
    return xx


if __name__ == '__main__':

    # open an image
    path = r'C:\Users\aayus\Desktop\NeuralStyleTransfer\parrot.jpg'
    img = image.load_img(path)

    # convert image to array and preprocess for vgg
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # we'll use this throughout the rest of the script
    batch_shape = x.shape
    shape = x.shape[1:]
    print(shape)

    # see the image
    plt.imshow(img)
    plt.show()

    # make a content model
    # try different cutoffs to see the images that result
    content_model = VGG16_AvgPool_CutOff(shape, 12)

    # make the target
    target = K.variable(content_model.predict(x))

    # try to match the image

    # define our loss in keras
    loss = K.mean(K.square(target - content_model.output))

    # gradients which are needed by the optimizer
    grads = K.gradients(loss, content_model.input)

    # just like theano.function
    get_loss_and_grads = K.function(
        inputs=[content_model.input],
        outputs=[loss] + grads
    )


    def get_loss_and_grads_wrapper(x_vec):
        # scipy's minimizer allows us to pass back
        # function value f(x) and its gradient f'(x)
        # simultaneously, rather than using the fprime arg
        #
        # we cannot use get_loss_and_grads() directly
        # input to minimizer func must be a 1-D array
        # input to get_loss_and_grads must be [batch_of_images]
        #
        # gradient must also be a 1-D array
        # and both loss and gradient must be np.float64
        # will get an error otherwise

        l0, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
        return l0.astype(np.float64), g.flatten().astype(np.float64)


    from datetime import datetime

    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    for i in range(30):
        x, l, _ = fmin_l_bfgs_b(
            func=get_loss_and_grads_wrapper,
            x0=x,
            # bounds=[[-127, 127]]*len(x.flatten()),
            maxfun=20
        )
        x = np.clip(x, -127, 127)
        # print("min:", x.min(), "max:", x.max())
        print("iter=%s, loss=%s" % (i+1, l))
        losses.append(l)

    print("duration:", datetime.now() - t0)
    plt.plot(losses)
    plt.show()

    new_img = x.reshape(*batch_shape)
    final_img = unpreprocess(new_img)

    plt.imshow(scale_img(final_img[0]))
    plt.show()

