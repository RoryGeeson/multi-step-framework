import tensorflow as tf
import dgl.nn.tensorflow as dglt
from numpy import prod

class multiNet(tf.keras.Model):
    """"""

    def __init__(self, stageGraph, h_dim, n_layers, network_type='GATConv', **kwargs):
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
            self.typeLayer[l].trainable = False
        for layer in self.typeLayer:
            _ = layer(stageGraph, tf.zeros([stageGraph.number_of_nodes(),self.h_dim]))

    def call(self, h_inputs, stageGraph):
        """"""
        h = tf.concat([h_inputs, stageGraph.ndata['objectives']], 1)
        for index, layer in enumerate(self.typeLayer):
            h = layer(stageGraph, h)
            if self.typeLayer[0].name == 'gat_conv':
                h = tf.reduce_mean(h, axis=1)
        return h

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

        self.preferenceEncoderNet = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1,self.objectiveNum)),
            tf.keras.layers.Dense(self.preferenceEncodeDim)
        ])

        self.hyperparameterGeneratorNet = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.numLayers,self.preferenceEncodeDim)),
            tf.keras.layers.Dense(self.hyperparameterGeneratorDim),
            tf.keras.layers.Dense(self.hyperparameterGeneratorDim),
            tf.keras.layers.Dense(self.weightsPerLayer, activation='tanh')
        ])


    def hyperparameterGenerator(self, preferenceEncoding):
        """"""
        preferenceEncodingTiled = tf.tile(preferenceEncoding, [self.numLayers,1])
        hyperparameters = self.hyperparameterGeneratorNet(preferenceEncodingTiled)
        return hyperparameters

    def call(self, objectiveWeightings):
        """"""
        preferenceEncoding = self.preferenceEncoderNet(objectiveWeightings)
        hyperparameters = self.hyperparameterGenerator(preferenceEncoding)

        return hyperparameters