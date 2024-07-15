import keras


class SAE2(keras.Model):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(input_dim,)),
                keras.layers.Dense(latent_dim[0], activation="sigmoid"),
                keras.layers.Dense(latent_dim[1], activation="sigmoid"),
            ]
        )
        self.decoder = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(latent_dim[1],)),
                keras.layers.Dense(latent_dim[0], activation="sigmoid"),
                keras.layers.Dense(input_dim, activation="sigmoid"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class SAE3(keras.Model):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(input_dim,)),
                keras.layers.Dense(latent_dim[0], activation="relu"),
                keras.layers.Dense(latent_dim[1], activation="relu"),
                keras.layers.Dense(latent_dim[2], activation="relu"),
            ]
        )
        self.decoder = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(latent_dim[2],)),
                keras.layers.Dense(latent_dim[1], activation="relu"),
                keras.layers.Dense(latent_dim[0], activation="relu"),
                keras.layers.Dense(input_dim, activation="relu"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)
