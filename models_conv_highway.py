import keras.backend as K
from keras import Model
from keras.layers import Dense, Multiply, Add, Lambda, Convolution2D,\
    Dropout, MaxPooling2D, Flatten
import keras.initializers


#https://github.com/keras-team/keras/issues/955
# https://gist.github.com/iskandr/a874e4cf358697037d14a17020304535
def conv_highway_layer(input, activation="tanh",kernel_size=3, num_filters=16, gate_bias=-3):
    # y = H(x, W_h)*T(x, W_t) + x*(1-T(x, W_t))
    # T(x, W_t, b_t) = sigmoid(W_t'*x+b_t)
    # H(x, W_h) = tanh(W_h'*x)
    # T = gate, H = regular layer, x = input, y = output
    dim = K.int_shape(input)[-1]
    gate = Convolution2D(dim, (kernel_size, kernel_size), padding='same', activation='sigmoid',
                         bias_initializer=keras.initializers.Constant(gate_bias))(input)  #T(x, W_t)
    H = Convolution2D(dim, (kernel_size, kernel_size), padding='same',
                                activation=activation)(input) #tanh(W_h'*x)
    negated_gate = Lambda(lambda x: 1.0 - x)(gate)
    transformed_gated = Multiply()([H, gate])  # H(x, W_h)*T(x, W_t)
    identity_gated = Multiply()([input, negated_gate])  #x*(1-T(x, W_t))
    output = Add()([transformed_gated, identity_gated])  # H(x, W_h)*T(x, W_t) + x*(1-T(x, W_t))
    return output




def half_and_half_layer(input, activation="tanh",kernel_size=3): # minimal glu
    # y = H(x, W_h) + x
    dim = K.int_shape(input)[-1]
    H = Convolution2D(dim, (kernel_size, kernel_size), padding='same',
                                activation=activation)(input) #tanh(W_h'*x)
    output_x2 = Add()([H, input])  # H(x, W_h)*T(x, W_t) + x*(1-T(x, W_t))
    output = Lambda(lambda x: x*.5)(output_x2)
    return output