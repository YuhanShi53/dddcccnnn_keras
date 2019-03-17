import tensorflow as tf
import keras
from keras import Sequential
from keras.models import Model
from keras.engine.topology import InputSpec
from keras.layers import Input, Dense, BatchNormalization, Layer
from keras.optimizers import Adam, SGD
import keras.backend as K

class Lenet(object):

    def __init__(self, emb_dim):
        self._emb_dim = emb_dim
        self.model = None

    def build_auto_encoder(self):
        input = Input(shape=(1024,), name="Input", dtype="float32")

        # Encoder
        encode_x = Dense(units=512, activation="relu")(input)
        encode_x = Dense(units=256, activation="relu")(encode_x)
        encode_x = Dense(units=128, activation="relu")(encode_x)
        encode_x = Dense(units=self._emb_dim, activation="relu")(encode_x)
        self.encoded = encode_x

        # Decoding
        decode_x = Dense(units=128, activation="relu")(encode_x)
        decode_x = Dense(units=256, activation="relu")(decode_x)
        decode_x = Dense(units=512, activation="relu")(decode_x)
        decode_x = Dense(units=1024, activation="relu")(decode_x)
        self.decoded = decode_x

        self.auto_encoder = Model(inputs=input, outputs=self.decoded, name="Auto_Encoder")
        self.encoder = Model(inputs=input, outputs=self.encoded, name="Encoder")

    def build_cluster(self):
        embedding = self.decoded


    def train_auto_encoder(self, X, y, batch_size, epochs, learning_rate=1e-4):
        self.auto_encoder.compile(optimizer=Adam(lr=learning_rate), loss="MSE")
        self.auto_encoder.fit(X, X, batch_size=batch_size, epochs=epochs, verbose=1)

    def train_cluster(self, X, y, batch_size, epochs, learnning_rate=1e-4):
        embedding = self.encoder.predict(X, batch_size=batch_size)


          

class ClusteringLayer(Layer):
    
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=128)

    def build(self, input_shape):
        input_dim = input_shape[1]
        # self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.centroids = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='centroids')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(ClusteringLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        cosine_similarity = self._cosine_similarity(inputs, self.centroids)
        ones = K.ones_like(cosine_similarity, dtype=float)
        cosine_distance = ones - cosine_similarity
        q = 1.0 / (1.0 + (K.sum(K.square(cosine_distance) / self.alpha)))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _cosine_similarity(self, X, centroids): 
        with tf.variable_scope("cosine_similarity"):
            numerator = K.dot(X, centroids) # shape=[batch_size, num_clusters]
            X_norm = K.l2_normalize(X, axis=1)
            centroids_norm = K.l2_normalize(centroids, axis=1)
            denominator = X_norm * centroids_norm
            cosine_similarity = numerator / denominator
            return cosine_similarity
