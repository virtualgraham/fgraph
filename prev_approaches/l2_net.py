import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, Lambda, Layer
import pickle
import numpy as np
import cv2

class LRN(Layer):

    def __init__(self, alpha=256,k=0,beta=0.5,n=256, **kwargs):
        super(LRN, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def call(self, x, mask=None):

            s = K.shape(x)
            b = s[0]
            r = s[1]
            c = s[2]
            ch = s[3]

            half_n = self.n // 2 # half the local region

            input_sqr = K.square(x) # square the input

            extra_channels = K.zeros((b, r, c, ch + 2 * half_n))
            input_sqr = K.concatenate([extra_channels[:, :, :, :half_n],input_sqr, extra_channels[:, :, :, half_n + ch:]], axis = 3)

            scale = self.k # offset for the scale
            norm_alpha = self.alpha / self.n # normalized alpha
            for i in range(self.n):
                scale += norm_alpha * input_sqr[:, :, :, i:i+ch]
            scale = scale ** self.beta
            x = x / scale
            
            return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def build_cnn(weights):

    model = Sequential()

    model.add(ZeroPadding2D(1, input_shape=(32,32, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(ZeroPadding2D(1))
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(ZeroPadding2D(1))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=2))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(ZeroPadding2D(1))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(ZeroPadding2D(1))
    model.add(Conv2D(128, kernel_size=(3, 3), strides=2))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(ZeroPadding2D(1))
    model.add(Conv2D(128, kernel_size=(3, 3)))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(Lambda(K.relu))

    model.add(Conv2D(128, kernel_size=(8, 8)))
    model.add(BatchNormalization(epsilon=0.0001, scale=False, center=False))
    model.add(LRN(alpha=256,k=0,beta=0.5,n=256))

    model.set_weights(weights)

    return model


def build_L2_net(net_name):

    python_net_data = pickle.load(open(net_name + ".pickle", "rb"))
    return build_cnn(python_net_data['weights']), build_cnn(python_net_data['weights_cen']), python_net_data['pix_mean'], python_net_data['pix_mean_cen']


def cal_L2Net_des(net_name, testPatchs, flagCS = False):

    """
    Get descriptors for one or more patches

    Parameters
    ----------
    net_name : string
        One of "L2Net-HP", "L2Net-HP+", "L2Net-LIB", "L2Net-LIB+", "L2Net-ND", "L2Net-ND+", "L2Net-YOS", "L2Net-YOS+",
    testPatchs : array
        A numpy array of image data with deimensions (?, 32, 32, 1), or if using central-surround with deimensions (?, 64, 64, 1)
    flagCS : boolean
        If True, use central-surround network

    Returns
    -------
    descriptor
        Numpy array with size (?, 128) or if using central-surround (?, 256)

    """

    model, model_cen, pix_mean, pix_mean_cen = build_L2_net(net_name)

    # print(model.summary())
    # print(model_cen.summary())

    if flagCS:

        testPatchsCen = testPatchs[:,16:48,16:48,:]
        testPatchsCen = testPatchsCen - pix_mean_cen
        testPatchsCen = np.array([(testPatchsCen[i] - np.mean(testPatchsCen[i]))/(np.std(testPatchsCen[i]) + 1e-12) for i in range(0, testPatchsCen.shape[0])])

        testPatchs = np.array([cv2.resize(testPatchs[i], (32,32), interpolation = cv2.INTER_CUBIC) for i in range(0, testPatchs.shape[0])])
        testPatchs = np.expand_dims(testPatchs, axis=-1)

    testPatchs = testPatchs - pix_mean
    testPatchs = np.array([(testPatchs[i] - np.mean(testPatchs[i]))/(np.std(testPatchs[i]) + 1e-12) for i in range(0, testPatchs.shape[0])])

    res = np.reshape(model.predict(testPatchs), (testPatchs.shape[0], 128))

    if flagCS:
        
        resCen = np.reshape(model_cen.predict(testPatchsCen), (testPatchs.shape[0], 128))

        return np.concatenate((res, resCen), 1)

    else:

        return res 


class L2Net:

    def __init__(self, net_name, flagCS = False):
        model, model_cen, pix_mean, pix_mean_cen = build_L2_net(net_name)
        self.flagCS = flagCS
        self.model = model
        self.model_cen = model_cen
        self.pix_mean = pix_mean
        self.pix_mean_cen = pix_mean_cen

    def calc_descriptors(self, patches):
        if self.flagCS:

            patchesCen = patches[:,16:48,16:48,:]
            patchesCen = patchesCen - self.pix_mean_cen
            patchesCen = np.array([(patchesCen[i] - np.mean(patchesCen[i]))/(np.std(patchesCen[i]) + 1e-12) for i in range(0, patchesCen.shape[0])])

            patches = np.array([cv2.resize(patches[i], (32,32), interpolation = cv2.INTER_CUBIC) for i in range(0, patches.shape[0])])
            patches = np.expand_dims(patches, axis=-1)

        patches = patches - self.pix_mean
        patches = np.array([(patches[i] - np.mean(patches[i]))/(np.std(patches[i]) + 1e-12) for i in range(0, patches.shape[0])])

        res = np.reshape(self.model.predict(patches), (patches.shape[0], 128))

        if self.flagCS:
            
            resCen = np.reshape(self.model_cen.predict(patchesCen), (patches.shape[0], 128))

            return np.concatenate((res, resCen), 1)

        else:

            return res 


# data = np.full((1,64,64,1), 0.)

# result = cal_L2Net_des("L2Net-HP+", data, flagCS=True)

# print(result)