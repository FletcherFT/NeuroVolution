from tensorflow import keras


class Agent:
    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Dense(100, activation=keras.layers.LeakyReLU(), name='LeakyReLu1', input_dim=6, use_bias=True),
            keras.layers.Dense(2, activation=keras.layers.LeakyReLU(), name='LeakyRelu2', use_bias=True)
        ])

    def step(self, inputs):
        return self.model.predict(inputs)
