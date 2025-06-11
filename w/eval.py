
from keras.models import Model
from keras.layers import Input, Dense

def create_model():
    net_input = Input(shape=(4,))
    x = Dense(64, activation="relu")(net_input)
    x = Dense(32, activation="relu")(x)
    output = Dense(2, activation="linear")(x)
    model = Model(inputs=net_input, outputs=output)
    model.compile(optimizer="adam", loss="mse")  # Optional compile
    return model


