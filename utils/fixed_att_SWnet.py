import keras.layers
import keras.models
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, \
    add, multiply
from keras.layers import concatenate, core, Dropout
from keras.models import Model
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers.core import Lambda
import keras.backend as K

CONST_DO_RATE = 0.5

option_dict_conv = {"activation": "relu", "padding": "same"}
option_dict_bn = {"momentum" : 0.9}

def attention_up_and_concate(down_layer, layer, data_format='channels_last'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]
    up = UpSampling2D(data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate


def attention_block_2d(x, g, inter_channel, data_format='channels_last'):

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    f = Activation('relu')(add([theta_x, phi_g]))

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)
    rate = Activation('sigmoid')(psi_f)
    att_x = multiply([x, rate])

    return att_x

def attention_3d_block_and_concate(down_layer):
    a_probs = tf.keras.layers.Dense(down_layer.get_shape().as_list()[3], activation='softmax')(down_layer)
    output_attention_mul = tf.keras.layers.multiply([down_layer, a_probs])
    return tf.keras.layers.concatenate([down_layer, output_attention_mul], axis=3)



# attention-enhanced simplified W-net
def get_core(dim1, dim2):
    
    x = tf.keras.layers.Input(shape=(dim1, dim2, 1))

     # DOWN 1
    a = tf.keras.layers.Convolution2D(64, 3, **option_dict_conv)(x)  
    a = tf.keras.layers.BatchNormalization(**option_dict_bn)(a)

    a = tf.keras.layers.Convolution2D(64, 3, **option_dict_conv)(a)
    a = tf.keras.layers.BatchNormalization(**option_dict_bn)(a)

    y = tf.keras.layers.MaxPooling2D()(a)

    b = tf.keras.layers.Convolution2D(128, 3, **option_dict_conv)(y)
    b = tf.keras.layers.BatchNormalization(**option_dict_bn)(b)

    b = tf.keras.layers.Convolution2D(128, 3, **option_dict_conv)(b)
    b = tf.keras.layers.BatchNormalization(**option_dict_bn)(b)

    y = tf.keras.layers.MaxPooling2D()(b)

    c = tf.keras.layers.Convolution2D(256, 3, **option_dict_conv)(y)
    c = tf.keras.layers.BatchNormalization(**option_dict_bn)(c)

    c = tf.keras.layers.Convolution2D(256, 3, **option_dict_conv)(c)
    c = tf.keras.layers.BatchNormalization(**option_dict_bn)(c)
    
    # UP 1
    
    c = tf.keras.layers.UpSampling2D()(c)
    c = attention_3d_block_and_concate(c)
    
    y = tf.keras.layers.concatenate([b, c], axis=3)

    d = tf.keras.layers.Convolution2D(128, 3, **option_dict_conv)(y)
    d = tf.keras.layers.BatchNormalization(**option_dict_bn)(d)

    d = tf.keras.layers.Convolution2D(128, 3, **option_dict_conv)(d)
    d = tf.keras.layers.BatchNormalization(**option_dict_bn)(d)

    # DOWN 2
    c1 = tf.keras.layers.MaxPooling2D()(c)
    d1 = tf.keras.layers.MaxPooling2D()(d)
    y = tf.keras.layers.concatenate([c1, d1], axis=3)

    e = tf.keras.layers.Convolution2D(256, 3, **option_dict_conv)(y)
    e = tf.keras.layers.BatchNormalization(**option_dict_bn)(e)

    e = tf.keras.layers.Convolution2D(256, 3, **option_dict_conv)(e)
    e = tf.keras.layers.BatchNormalization(**option_dict_bn)(e)

    # UP 2
    e = tf.keras.layers.UpSampling2D()(e)
    e = attention_3d_block_and_concate(e)
    y = tf.keras.layers.concatenate([d, e], axis=3)

    f = tf.keras.layers.Convolution2D(128, 3, **option_dict_conv)(y)
    f = tf.keras.layers.BatchNormalization(**option_dict_bn)(f)

    f = tf.keras.layers.Convolution2D(128, 3, **option_dict_conv)(f)
    f = tf.keras.layers.BatchNormalization(**option_dict_bn)(f)

    f = tf.keras.layers.UpSampling2D()(f)
    f = attention_3d_block_and_concate(f)
    
    y = tf.keras.layers.concatenate([f, a], axis=3)

    y = tf.keras.layers.Convolution2D(64, 3, **option_dict_conv)(y)
    y = tf.keras.layers.BatchNormalization(**option_dict_bn)(y)

    y = tf.keras.layers.Convolution2D(64, 3, **option_dict_conv)(y)
    y = tf.keras.layers.BatchNormalization(**option_dict_bn)(y)

    return [x, y]


def get_model_3_class(dim1, dim2, activation="softmax"):
    
    [x, y] = get_core(dim1, dim2)

    y = tf.keras.layers.Convolution2D(3, 1, **option_dict_conv)(y)

    if activation is not None:
        y = tf.keras.layers.Activation(activation)(y)

    model = tf.keras.models.Model(x, y)
    
    return model


