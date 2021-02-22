import dgl
import tensorflow as tf

# def trainTestSplitter():
#     """"""

def parameterizeModel(model, weights):
    """
    based off:
    https://github.com/hummosa/Hypernetworks_Keras_TF2/blob/master/Hypernetworks_in_keras_and_tf2.ipynb Remember to properly credit
    :param model:
    :param weights:
    :return:
    """
    weights = tf.reshape(weights, [-1])  # reshape the parameters to a vector

    last_used = 0
    for i, layer in enumerate(model.layers):
        # check to make sure only conv and fully connected layers are assigned weights.
        if 'dense' in layer.name:
            weights_shape = layer.kernel.shape
            no_of_weights = tf.reduce_prod(weights_shape)
            new_weights = tf.reshape(weights[last_used:last_used + no_of_weights], weights_shape)
            layer.kernel = new_weights
            last_used += no_of_weights

            if layer.use_bias:
                weights_shape = layer.bias.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used + no_of_weights], weights_shape)
                layer.bias = new_weights
                last_used += no_of_weights

def determineNumVariables(model):
    """"""
    numVariables = model.trainable_variables
    return numVariables

class objective():

    def __init__(self, objectiveFunction, objectiveVariables):
        self.objectiveFunction = objectiveFunction
        self.objectiveVariables = objectiveVariables

class stage():

    def __init__(self, stageID, model, inputRanges, followingStages=None):
        self.stageID = stageID
        self.model = model
        self.inputRanges = inputRanges
        # followingStages includes the stageID and the variables relevant to it
        self.followingStages = followingStages

def createHyperparameterLenDict(model):
    hyperparameterLenDict = {}
    for index, layer in enumerate(model.typeLayer):
        hyperparameterLenDict['layer{}'.format(index)] = []
        for weight in layer.trainable_weights:
             hyperparameterLenDict['layer{}'.format(index)].append(weight.shape)
    return hyperparameterLenDict