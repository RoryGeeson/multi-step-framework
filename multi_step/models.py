import tensorflow as tf
import dgl.nn.tensorflow as dglt
from numpy import prod

class multiNet(tf.keras.Model):
    """"""

    def __init__(self, stageGraph, h_dim, n_layers, trainable=False, network_type='GATConv', **kwargs):
        super().__init__()

        network_type_dict = {
            'GraphConv': dglt.GraphConv,
            'RelGraphConv': dglt.RelGraphConv,
            'GATConv': dglt.GATConv,
            'SAGEConv': dglt.SAGEConv,
            'SGConv': dglt.SGConv,
            'APPNPConv': dglt.APPNPConv,
            'GINConv': dglt.GINConv
        }

        self.h_dim = h_dim + 1
        self.n_layers = n_layers

        self.typeLayer = []

        for l in range(n_layers):
            self.typeLayer.append(network_type_dict[network_type](self.h_dim, self.h_dim, **kwargs))
            self.typeLayer[l].trainable = trainable
        for layer in self.typeLayer:
            _ = layer(stageGraph, tf.zeros([stageGraph.number_of_nodes(),self.h_dim]))

    def call(self, h_inputs, stageGraph):
        """"""
        h = tf.concat([h_inputs, stageGraph.ndata['objectives']], 1)
        print(h)
        for index, layer in enumerate(self.typeLayer):
            h = layer(stageGraph, h)
            if layer.name[0:3] == 'gat':
                h = tf.reduce_mean(h, axis=1)
        return tf.squeeze(h)

class multiHyperNet(tf.keras.Model):
    """"""

    def __init__(self, numObjectives, preferenceEncodeDim, hyperparameterGeneratorDim, hyperparameterLenDict):
        super().__init__()

        self.objectiveNum = numObjectives  # Set the number of objectives to be balanced
        self.preferenceEncodeDim = preferenceEncodeDim  # Set the dimension of the hidden preference vector
        self.hyperparameterGeneratorDim = hyperparameterGeneratorDim
        self.hyperparameterLenDict = hyperparameterLenDict
        self.numLayers = len(self.hyperparameterLenDict.keys())
        self.weightsPerLayer = sum(
            [prod(layerWeights) for layerWeights in list(self.hyperparameterLenDict.values())[0]])

        # self.preferenceEncoderNet = tf.keras.Sequential([
        #     tf.keras.layers.InputLayer(input_shape=(1,self.objectiveNum)),
        #     tf.keras.layers.Dense(self.preferenceEncodeDim)
        # ])

        # Navon paper preference encoder
        # nn.Linear(2, ray_hidden_dim),
        # nn.ReLU(inplace=True),
        # nn.Linear(ray_hidden_dim, ray_hidden_dim),
        # nn.ReLU(inplace=True),
        # nn.Linear(ray_hidden_dim, ray_hidden_dim)

        self.preferenceEncoderNet = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1, self.objectiveNum)),
            tf.keras.layers.Dense(self.preferenceEncodeDim * self.numLayers, activation='relu'),
            # tf.keras.layers.Dense(self.preferenceEncodeDim * self.numLayers, activation='relu')
        ])

        self.hyperparameterGeneratorNet = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.numLayers,self.preferenceEncodeDim)),
            # tf.keras.layers.Dense(self.hyperparameterGeneratorDim, activation='softmax'),
            tf.keras.layers.Dense(self.hyperparameterGeneratorDim, activation='relu'),
            # tf.keras.layers.Dense(self.weightsPerLayer, activation='tanh'),
            tf.keras.layers.Dense(self.weightsPerLayer)
        ])


    def hyperparameterGenerator(self, preferenceEncoding):
        """"""
        # preferenceEncodingTiled = tf.tile(preferenceEncoding, [self.numLayers,1])
        # hyperparameters = self.hyperparameterGeneratorNet(preferenceEncodingTiled)
        preferenceEncoding = tf.reshape(preferenceEncoding, [self.numLayers, -1])
        hyperparameters = self.hyperparameterGeneratorNet(preferenceEncoding)
        return hyperparameters

    def call(self, objectiveWeightings):
        """"""
        preferenceEncoding = tf.reshape(self.preferenceEncoderNet(objectiveWeightings),[1,-1])
        hyperparameters = self.hyperparameterGenerator(preferenceEncoding)

        return hyperparameters

class stageEncoder(tf.keras.Model):
    def __init__(self, inputLength, h_dim):
        super().__init__()

        self.encoderNet = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1, inputLength)),
            tf.keras.layers.Dense(h_dim, activation='tanh'),
            tf.keras.layers.Dense(h_dim)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.encoderNet(inputs)


class stageDecoder(tf.keras.Model):
    def __init__(self, inputLength, h_dim):
        super().__init__()

        self.encoderNet = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1, h_dim + 1)),
            # tf.keras.layers.Dense(inputLength, activation='tanh')
            tf.keras.layers.Dense(inputLength, activation='exponential'),
            tf.keras.layers.Dense(inputLength)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.encoderNet(inputs)

class objectiveDecoder(tf.keras.Model):
    def __init__(self, numObjectives, numStages, h_dim):
        super().__init__()

        self.encoderNet = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(numStages*h_dim,)),
            tf.keras.layers.Dense(numObjectives, activation='exponential'),
            tf.keras.layers.Dense(numObjectives)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.encoderNet(inputs)

class multiNetPredict(tf.keras.Model):
    """"""

    def __init__(self, stageGraph, h_dim, n_layers, trainable=True, network_type='GATConv',
                 **kwargs):
        super().__init__()

        network_type_dict = {
            'GraphConv': dglt.GraphConv,
            'RelGraphConv': dglt.RelGraphConv,
            'GATConv': dglt.GATConv,
            'SAGEConv': dglt.SAGEConv,
            'SGConv': dglt.SGConv,
            'APPNPConv': dglt.APPNPConv,
            'GINConv': dglt.GINConv
        }

        self.h_dim = h_dim
        self.n_layers = n_layers

        self.typeLayer = []

        for l in range(n_layers):
            self.typeLayer.append(network_type_dict[network_type](self.h_dim, self.h_dim, **kwargs))
            self.typeLayer[l].trainable = trainable
        for layer in self.typeLayer:
            _ = layer(stageGraph, tf.zeros([stageGraph.number_of_nodes(), self.h_dim]))

    def call(self, h_inputs, stageGraph):
        """"""
        # h = tf.concat([h_inputs, stageGraph.ndata['objectives']], 1)
        for index, layer in enumerate(self.typeLayer):
            h = layer(stageGraph, h_inputs)
            if layer.name[0:3] == 'gat':
                h = tf.reduce_mean(h, axis=1)
        return tf.squeeze(h)