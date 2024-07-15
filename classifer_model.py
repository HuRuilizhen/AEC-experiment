import keras


class Softmax_Classifer(keras.Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.classifier = keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(input_dim,)),
                keras.layers.Dense(output_dim, activation="softmax"),
            ]
        )

    def call(self, x):
        return self.classifier(x)


from sklearn.svm import SVC


class SVM_Classifer(keras.Model):
    def __init__(self):
        super().__init__()
        self.svm = SVC(
            kernel="poly",
            degree=2,
            C=1.0,
            gamma="auto",
        )

    def train(self, x, y):
        self.svm.fit(x, y)

    def call(self, x):
        return self.svm.predict(x)
