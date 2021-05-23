import keras.layers
import keras.models
import tensorflow as tf

CONST_DO_RATE = 0.5

option_dict_conv = {"activation": "relu", "padding": "same"}
option_dict_bn = {"momentum" : 0.9}


# returns a core model from gray input to 64 channels of the same size
def get_core(dim1, dim2):
    
    x = tf.keras.layers.Input(shape=(dim1, dim2, 1))

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

    
    y = tf.keras.layers.MaxPooling2D()(c)

    d = tf.keras.layers.Convolution2D(512, 3, **option_dict_conv)(y)
    d = tf.keras.layers.BatchNormalization(**option_dict_bn)(d)

    d = tf.keras.layers.Convolution2D(512, 3, **option_dict_conv)(d)
    d = tf.keras.layers.BatchNormalization(**option_dict_bn)(d)

    
    # UP

    d = tf.keras.layers.UpSampling2D()(d)

    y = tf.keras.layers.concatenate([d, c], axis=3)

    e = tf.keras.layers.Convolution2D(256, 3, **option_dict_conv)(y)
    e = tf.keras.layers.BatchNormalization(**option_dict_bn)(e)

    e = tf.keras.layers.Convolution2D(256, 3, **option_dict_conv)(e)
    e = tf.keras.layers.BatchNormalization(**option_dict_bn)(e)

    e = tf.keras.layers.UpSampling2D()(e)

    
    y = tf.keras.layers.concatenate([e, b], axis=3)

    f = tf.keras.layers.Convolution2D(128, 3, **option_dict_conv)(y)
    f = tf.keras.layers.BatchNormalization(**option_dict_bn)(f)

    f = tf.keras.layers.Convolution2D(128, 3, **option_dict_conv)(f)
    f = tf.keras.layers.BatchNormalization(**option_dict_bn)(f)

    f = tf.keras.layers.UpSampling2D()(f)

    
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
