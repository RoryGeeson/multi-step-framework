import tensorflow as tf

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
        for weight in layer.weights:
            hyperparameterLenDict['layer{}'.format(index)].append(weight.shape)
    return hyperparameterLenDict

def updateHyperparameters(hyperparameterLenDict, hyperparameters, model):
    for index, layer in enumerate(model.typeLayer):
        layerWeights = hyperparameterLenDict['layer{}'.format(index)]
        if 'gat_conv' in layer.name:
            startInd = 0
            endInd = startInd + tf.reduce_prod(layerWeights[0])
            layer.attn_l = tf.reshape(hyperparameters[index,startInd:endInd], layerWeights[0])
            startInd = endInd
            endInd = startInd + tf.reduce_prod(layerWeights[1])
            layer.attn_r = tf.reshape(hyperparameters[index,startInd:endInd], layerWeights[1])
            startInd = endInd
            endInd = startInd + tf.reduce_prod(layerWeights[2])
            layer.fc.kernel = tf.reshape(hyperparameters[index,startInd:endInd], layerWeights[2])

def scaleInputs(inputs, inputRanges):
    scaledInputs = tf.math.multiply(inputs, inputRanges[:,0]) + inputRanges[:,1]
    return scaledInputs