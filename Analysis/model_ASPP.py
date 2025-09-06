import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def expend_as(tensor, rep):
    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                  arguments={'repnum': rep})(tensor)


def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :param input:   down-dim feature map
    :param out_size:output channel number
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = nn.Conv2d(out_size[0], out_size[1], (1, 1), padding='same')(input)
    if batch_norm:
        x = nn.BatchNorm2d(x)
    x = nn.ReLU(x)
    return x


def attention_block(x, gating, inter_shape):
    shape_x = x.shape
    shape_g = gating.shape

    theta_x = nn.Conv2d(inter_shape[0], inter_shape[1], (2, 2), stride=2, padding='same')(x)  # 16
    shape_theta_x = theta_x.shape

    phi_g = nn.Conv2d(inter_shape[0],inter_shape[1] , (1, 1), padding='same')(gating)
    upsample_g = nn.ConvTranspose2d(inter_shape[0], inter_shape[1], (3, 3),
                                 stride=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding_mode='same')(phi_g)  # 16

    concat_xg = torch.cat([upsample_g, theta_x])
    act_xg = nn.ReLU(concat_xg)
    psi = nn.Conv2d(1, 1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = nn.Sigmoid(psi)
    shape_sigmoid = sigmoid_xg.shape
    upsample_psi = nn.Upsample(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[3])

    y = multiply([upsample_psi, x])

    result = nn.Conv2d(shape_x[3],shape_x[3], (1, 1), padding='same')(y)
    result_bn = nn.BatchNorm2d(result)
    return result_bn


def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = nn.AvgPool2d(init)
    se = se_shape.reshape(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x


## regular conv block

# def conv_block(inputs, filters, drop_out=0.0):
#     x = inputs

#     x = Conv2D(filters, (3, 3), padding="same")(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)

#     x = Conv2D(filters, (3, 3), padding="same")(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)

#     if drop_out > 0:
#         x = Dropout(drop_out)(x)

#     x = squeeze_excite_block(x)

#     return x

## residual conv block

def conv_block(inputs, filters, drop_out=0.0):
    x = inputs
    shortcut = inputs

    x = nn.Conv2d(filters, (3, 3), padding="same")(x)
    x = nn.BatchNorm2d(x)
    x = nn.ReLU(x)

    x = nn.Conv2d(filters, (3, 3), padding="same")(x)
    x = nn.BatchNorm2d(x)

    shortcut = nn.Conv2d(filters, (1, 1), padding="same")(shortcut)
    shortcut = nn.BatchNorm2d(shortcut)

    x = torch.cat([shortcut, x])
    x = nn.ReLU(x)

    if drop_out > 0:
        x = nn.Dropout(drop_out)(x)

    x = squeeze_excite_block(x)

    return x


def encoder1(inputs):
    skip_connections = []

    model = VGG19(include_top=False, weights='imagenet', input_tensor=inputs)
    names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
    for name in names:
        skip_connections.append(model.get_layer(name).output)

    output = model.get_layer("block5_conv4").output
    return output, skip_connections


def decoder1(inputs, skip_connections):
    num_filters = [256, 128, 64, 32]
    ##custom code##{
    channels = [512, 256, 128, 64]
    ##custom code##}
    skip_connections.reverse()
    x = inputs
    shape = x.shape

    for i, f in enumerate(num_filters):
        ##custom code##{
        gating = gating_signal(x, channels[i], True)
        att = attention_block(skip_connections[i], gating, channels[i])
        ##custom code##}
        x = nn.ConvTranspose2d(shape[3], (2, 2), activation="relu", strides=(2, 2))(x)
        # x = Concatenate()([x, skip_connections[i]])
        ##custom code##{
        x = torch.concat([x, att])
        ##custom code##}

        print(f"Applying dropout in decoder1 up layer {i + 1}")
        if i < 2:
            x = conv_block(x, f, drop_out=0.5)
        else:
            x = conv_block(x, f, drop_out=0.3)

    return x


def encoder2(inputs):
    num_filters = [32, 64, 128, 256]
    skip_connections = []
    x = inputs

    for i, f in enumerate(num_filters):
        x = conv_block(x, f)
        skip_connections.append(x)
        x = nn.MaxPool2d((2, 2))(x)

    return x, skip_connections


def decoder2(inputs, skip_1, skip_2):
    num_filters = [256, 128, 64, 32]
    ##custom code##{
    channels = [512, 256, 128, 64]
    ##custom code##}
    skip_2.reverse()
    x = inputs

    for i, f in enumerate(num_filters):
        ##custom code##{
        gating_enc_2 = gating_signal(x, num_filters[i], True)
        att_enc_2 = attention_block(skip_2[i], gating_enc_2, num_filters[i])
        ##custom code##}

        x = nn.Upsample((2, 2))(x)
        # x = Concatenate()([x, skip_1[i], skip_2[i]])
        ##custom code##{
        x = torch.concat([x, skip_1[i], att_enc_2])
        ##custom code##}

        print(f"Applying dropout in decoder2 up layer {i + 1}")
        if i < 2:
            x = conv_block(x, f, drop_out=0.5)
        else:
            x = conv_block(x, f, drop_out=0.5)

    return x


def output_block(inputs):
    x = nn.Conv2d(1, (1, 1), padding_mode="same")(inputs)
    x = nn.Sigmoid(x)
    return x


def Upsample(tensor, size):
    """Bilinear upsampling"""

    def _upsample(x, size):
        return tf.image.resize(images=x, size=size)

    return Lambda(lambda x: _upsample(x, size), output_shape=size)(tensor)


def ASPP(x, filter):
    shape = x.shape

    y1 = nn.AvgPool2d((shape[1], shape[2]))(x)
    y1 = nn.Conv2d(filter, 1, padding_mode="same")(y1)
    y1 = nn.BatchNorm2d(y1)
    y1 = nn.ReLU(y1)
    y1 = nn.Upsample((shape[1], shape[2]))(y1)

    y2 = nn.Conv2d(filter, 1, dilation=1, padding="same", bias=False)(x)
    y2 = nn.BatchNorm2d(y2)
    y2 = nn.ReLU(y2)

    y3 = nn.Conv2d(filter, 3, dilation=6, padding="same", use_bias=False)(x)
    y3 = nn.BatchNorm2d(y3)
    y3 = nn.ReLU(y3)

    y4 = nn.Conv2d(filter, 3, dilation=12, padding="same", bias=False)(x)
    y4 = nn.BatchNorm2d(y4)
    y4 = nn.ReLU(y4)

    y5 = nn.Conv2d(filter, 3, dilation=18, padding="same", bias=False)(x)
    y5 = nn.BatchNorm2d(y5)
    y5 = nn.ReLU(y5)

    y = torch.concat([y1, y2, y3, y4, y5])

    y = nn.Conv2d(filter, 1, dilation=1, padding="same", bias=False)(y)
    y = nn.BatchNorm2d(y)
    y = nn.ReLU(y)

    return y


def build_model_ASPP(shape):
    inputs = Input(shape)
    x, skip_1 = encoder1(inputs)
    x = ASPP(x, 64)
    x = decoder1(x, skip_1)
    outputs1 = output_block(x)

    x = inputs * outputs1

    x, skip_2 = encoder2(x)
    x = ASPP(x, 64)
    x = decoder2(x, skip_1, skip_2)
    outputs2 = output_block(x)
    outputs = Concatenate()([outputs1, outputs2])

    model = Model(inputs, outputs)
    return model