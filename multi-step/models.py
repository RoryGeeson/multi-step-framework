import tensorflow as tf
import dgl
import dgl.nn.tensorflow as dglt
from utils import parameterizeModel
# DGLBACKEND=tensorflow

# class multiNetEncoder(tf.keras.layers.Layer):
#     """"""
#
#     def __init__(self, h_dim):
#         super().__init__()
#
#         self.encoder = tf.keras.models.Sequential([
#             tf.keras.layers.InputLayer(),
#             tf.keras.layers.Dense(h_dim)
#         ])
#
#     def call(self, input):
#         encodedInput = self.encoder(input)
#         return encodedInput
#
# class multiNetDecoder(tf.keras.layers.Layer):
#     """"""
#
#     def __init__(self, h_dim):
#         super().__init__()
#
#         self.layerModel = tf.keras.Sequential([
#             tf.keras.layers.InputLayer(h_dim),
#             tf.keras.layers.Dense()
#         ])
#
#     def call(self, h):
#         output = self.layerModel(h)
#         return output


class multiNet(tf.keras.Model):
    """"""

    def __init__(self, stageGraph, h_dim, n_layers, network_type='GATConv', **kwargs):
        super().__init__()

        network_type_dict = {
            'GraphConv': dglt.GraphConv,
            'RelGraphConv': dglt.RelGraphConv,
            # 'GATConv': dglt.GATConv(h_dim, h_dim, **kwargs),
            'GATConv': dglt.GATConv,
            'SAGEConv': dglt.SAGEConv,
            'SGConv': dglt.SGConv,
            'APPNPConv': dglt.APPNPConv,
            'GINConv': dglt.GINConv
        }

        self.stageGraph = stageGraph

        self.h_dim = h_dim + 1
        self.n_layers = n_layers

        # self.encoder = multiNetEncoder(h_dim)
        # self.decoder = multiNetDecoder(h_dim)

        self.typeLayer = []
        self.hidden_layer = network_type_dict[network_type](h_dim, h_dim, **kwargs)
        # self.hidden_layer.__init__(self=self.hidden_layer, in_feats=h_dim, out_feats=h_dim, **kwargs)
        for l in range(n_layers):
            self.typeLayer.append(self.hidden_layer)

    def call(self, stageGraph, hyperparametersDict):
        """"""
        for stageID in stageGraph.stageIDs:
            stageGraph.nodes[stageID].data['h_inputs'] = stageGraph.stageEncoders[stageID](tf.expand_dims(stageGraph.stageInputs[stageID],0))
        self.typeLayer[0].set_weights(hyperparametersDict['layer0'])
        print('h_inputs', stageGraph.ndata['h_inputs'])
        print('objectives: ',stageGraph.ndata['objectives'])
        print('concat: ',tf.concat([stageGraph.ndata['h_inputs'], stageGraph.ndata['objectives']], 1))
        stageGraph.ndata['h_inputs_objectives'] = tf.concat([stageGraph.ndata['h_inputs'], stageGraph.ndata['objectives']], 1)
        h = self.typeLayer[0](stageGraph, stageGraph.ndata['h_inputs_objectives'])
        for index, layer in enumerate(self.typeLayer[1:]):
            layer.set_weights(hyperparametersDict['layer{}'.format(index + 1)])
            h = layer(stageGraph, h)
        # output = self.decoder(h, weights=hyperParameters['decoder'])
        # for stageID in stageGraph.stageIDs:
        #     output = stageGraph.stageDecoders[stageID](h[stageID,:])
        for stageID in stageGraph.stageIDs:
            stageGraph.nodes[stageID].data['generated'] = stageGraph.stageDecoders[stageID](h[stageID,:])
        # loss = stageGraph.objectiveFunctions(output)
        # return loss

class multiHyperNet(tf.keras.Model):
    """"""

    def __init__(self, numObjectives, preferenceEncodeDim, hyperparameterGeneratorDim, hyperparameterLenDict):
        super().__init__()

        self.objectiveNum = numObjectives  # Set the number of objectives to be balanced
        self.preferenceEncodeDim = preferenceEncodeDim  # Set the dimension of the hidden preference vector
        self.hyperparameterGeneratorDim = hyperparameterGeneratorDim
        self.hyperparameterLenDict = hyperparameterLenDict
        self.numLayers = len(self.hyperparameterLenDict.keys())
        self.weightsPerLayer = len(list(self.hyperparameterLenDict.values())[0])
        self.hyperparameterNum = tf.reduce_prod(list(self.hyperparameterLenDict.values())[0][0])*self.weightsPerLayer

        self.preferenceEncoderNet = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1,self.objectiveNum)),
            # tf.keras.layers.Dense(self.preferenceEncodeDim, activation= 'relu'),
            tf.keras.layers.Dense(self.preferenceEncodeDim)
        ])

        self.hyperparameterGeneratorNet = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.numLayers,self.preferenceEncodeDim)),
            # tf.keras.layers.Dense(self.hyperparameterGeneratorDim),
            tf.keras.layers.Dense(self.hyperparameterNum, activation='tanh')
        ])

    # def preferenceEncoder(self, preferenceVector):
    #     """"""
    #     encodedPreference = self.preferenceEncoderNet(preferenceVector)
    #     return encodedPreference

    def hyperparameterGenerator(self, preferenceEncoding):
        """"""

        preferenceEncodingTiled = tf.tile(preferenceEncoding, [self.numLayers,1])
        hyperparameters = self.hyperparameterGeneratorNet(preferenceEncodingTiled)

        # Create dictionary for the hyperparameters
        hyperparametersDict = {}
        for layerNum, key in enumerate(self.hyperparameterLenDict.keys()):
            hyperparametersDict[key] = []
            layerWeights = tf.reshape(hyperparameters[layerNum], [self.weightsPerLayer, -1])
            for weight in range(self.weightsPerLayer):
                hyperparametersDict[key].append(tf.reshape(layerWeights[weight], self.hyperparameterLenDict[key][weight]))
        return hyperparametersDict

    def call(self, objectiveWeightings):
        """"""
        preferenceEncoding = self.preferenceEncoderNet(objectiveWeightings)
        hyperparametersDict = self.hyperparameterGenerator(preferenceEncoding)

        return hyperparametersDict