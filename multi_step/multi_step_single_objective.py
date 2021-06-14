# import tensorflow as tf
from .utils import *
from .models import *
import dgl
import tensorflow.keras.optimizers as optimizers
import numpy as np
from math import inf

def stageGraphDecorator(graph):
    setattr(graph, 'stages', {})
    setattr(graph, 'stageIDs', [])
    setattr(graph, 'stageInputs', {})
    setattr(graph, 'stageOutputs', {})
    setattr(graph, 'stageGeneratedInputs', {})
    setattr(graph, 'stagePrevious', {})
    # setattr(graph, 'stageParameters', {})
    setattr(graph, 'stageInputRanges', {})
    setattr(graph, 'stageInputLength', {})
    setattr(graph, 'stageOutputLength', {})
    setattr(graph, 'stageOutput', {})
    setattr(graph, 'stageEncoders', {})
    setattr(graph, 'stageDecoders', {})

    def loadStages(self, stages):
        for stage in stages:
            self.stageIDs.append(stage.stageID)
            # self.stageParameters[stage.stageID] = stage.model.trainable_variables
            self.stages[stage.stageID] = stage.model

            inputRanges = np.array(stage.inputRanges)
            self.stageInputRanges[stage.stageID] = np.vstack((0.5*(inputRanges[:,1]-inputRanges[:,0]), np.average(inputRanges, axis=1))).T

            self.stageInputLength[stage.stageID] = len(stage.inputRanges)
            if not stage.outputDimension:
                self.stageOutputLength[stage.stageID] = stage.model.y_dim
            else:
                self.stageOutputLength[stage.stageID] = stage.outputDimension

    setattr(graph, 'loadStages', loadStages)

    def defineConnectivity(self, stages):
        for stage in stages:
            if not stage.followingStages == None:
                for followingStage in stage.followingStages:
                    # self.stageInputLength[followingStage[0]] -= len(followingStage[1])
                    self.add_edges(stage.stageID, followingStage[0])
                    try:
                        self.stagePrevious[followingStage[0]].append([stage.stageID, followingStage[1]])
                    except:
                        self.stagePrevious[followingStage[0]] = [[stage.stageID, followingStage[1]]]

    setattr(graph, 'defineConnectivity', defineConnectivity)

    def calculateOutputs(self, nodeDataIdentifier):
        for stageID in self.stageIDs:
            try:
                previousStageOutputs = []
                for previousStage in self.stagePrevious[stageID]:
                    tempPreviousStage = self.stageOutputs[previousStage[0]]
                    for stageVar in previousStage[1]:
                        if tempPreviousStage.shape == ():
                            previousStageOutputs.append(tempPreviousStage)
                        else:
                            previousStageOutputs.append(tempPreviousStage[stageVar])
                if nodeDataIdentifier == 'inputs':
                    modelInputs = tf.concat([previousStageOutputs, self.stageInputs[stageID]], 0)
                elif nodeDataIdentifier == 'stageGeneratedInputs':
                    modelInputs = tf.concat([previousStageOutputs, self.stageGeneratedInputs[stageID]], 0)
                else:
                    modelInputs = tf.concat([previousStageOutputs, self.stageInputs[stageID]], 0)
            except:
                if nodeDataIdentifier == 'inputs':
                    modelInputs = self.stageInputs[stageID]
                elif nodeDataIdentifier == 'stageGeneratedInputs':
                    modelInputs = self.stageGeneratedInputs[stageID]
                else:
                    modelInputs = self.stageInputs[stageID]
            if self.stages[stageID].name == 'gpr':
                try:
                    output = tf.squeeze(self.stages[stageID].predict_f(Xnew=tf.expand_dims(modelInputs,0))[0])
                except:
                    output = tf.squeeze(self.stages[stageID].predict_f_compiled(Xnew=tf.expand_dims(modelInputs, 0))[0])
            else:
                output = self.stages[stageID].evaluateModel(modelInputs)
            self.stageOutputs[stageID] = output

    setattr(graph, 'calculateOutputs', calculateOutputs)

    def applyInputs(self, inputs):
        endIndex = 0
        for stageID in self.stageIDs:
            startIndex = endIndex
            endIndex = endIndex + self.stageInputLength[stageID]
            self.stageInputs[stageID] = inputs[startIndex:endIndex]

    setattr(graph, 'applyInputs', applyInputs)

    return graph

class objectiveFunctions():
    """"""

    def __init__(self, objectives, stageGraph):
        self.objectiveIDs = []
        self.objectives = {}
        self.objectiveScalings = {}
        self.objectiveVariableIDs = {}
        self.defineObjectives(objectives, stageGraph)

    def defineObjectives(self, objectives, stageGraph):
        stageGraph.add_nodes(len(objectives), {'objectives': tf.zeros([len(objectives), 1])})
        num_stages = len(stageGraph.stageIDs)
        for objectiveIndex, objective in enumerate(objectives):
            objectiveID = objectiveIndex + num_stages
            self.objectiveIDs.append(objectiveID)
            self.objectives[objectiveID] = objective.objectiveFunction
            self.objectiveScalings[objectiveID] = -inf
            self.objectiveVariableIDs[objectiveID] = objective.objectiveVariables
            stageGraph.add_edges([objectiveID]*len(stageGraph.stageIDs), stageGraph.stageIDs)

    def normaliseObjective(self, objectiveValue, objectiveID):
        return objectiveValue

    def calculateObjective(self, stageGraph, objectiveID, nodeDataIdentifier='inputs'):
        objectiveVariableValues = []
        for variable in self.objectiveVariableIDs[objectiveID]:
            variableOffset = calculateOffset(stageGraph, variable[0].stageID)
            if variable[1] == 'inputs':
                if nodeDataIdentifier == 'inputs':
                    try:
                        objectiveVariableValues.append(
                            stageGraph.stageInputs[variable[0].stageID][variable[2] - variableOffset])
                    except:
                        objectiveVariableValues.append(stageGraph.stageInputs[variable[0].stageID])
                else:
                    try:
                        objectiveVariableValues.append(
                            stageGraph.stageGeneratedInputs[variable[0].stageID][variable[2] - variableOffset])
                    except:
                        objectiveVariableValues.append(stageGraph.stageGeneratedInputs[variable[0].stageID])
            else:
                try:
                    objectiveVariableValues.append(
                        stageGraph.stageOutputs[variable[0].stageID][variable[2] - variableOffset])
                except:
                    objectiveVariableValues.append(stageGraph.stageOutputs[variable[0].stageID])
        objectiveValue = tf.reshape(self.objectives[objectiveID](*objectiveVariableValues), [1, 1])
        return objectiveValue

    def calculateObjectives(self, stageGraph):
        for ID in self.objectiveIDs:
            objectiveValueUnnorm = self.calculateObjective(stageGraph, ID)
            stageGraph.nodes[ID].data['objectives'] = self.normaliseObjective(objectiveValueUnnorm, ID)

    def calculateWeightedObjective(self, stageGraph, objectiveWeightings, nodeDataIdentifier):
        """"""
        objectiveValue = 0
        for index, ID in enumerate(self.objectiveIDs):
            objectiveValueUnnorm = self.calculateObjective(stageGraph, ID, nodeDataIdentifier)
            objectiveValue += self.normaliseObjective(objectiveValueUnnorm,ID) * objectiveWeightings[0, index]
        return objectiveValue

    def calculateUnnormalisedLosses(self, stageGraph):
        objectives = self.calculateObjective(stageGraph, self.objectiveIDs[0], 'stageGeneratedInputs')
        if len(self.objectiveIDs) > 1:
            for ID in self.objectiveIDs[1:]:
                obj = self.calculateObjective(stageGraph, ID, 'stageGeneratedInputs')
                objectives = tf.concat([objectives, obj], axis=0)
        return objectives

    def unnormalisedObjectives(self, stageGraph):
        objectiveList = []
        for ID in self.objectiveIDs:
            objectiveValue = self.calculateObjective(stageGraph, ID, 'stageGeneratedInputs')
            objectiveList.append(objectiveValue.numpy())
        return np.squeeze(np.array(objectiveList))

    def resetScalings(self):
        for scalingKey in self.objectiveScalings.keys():
            self.objectiveScalings[scalingKey] = -inf

class multi_step_graph():
    """"""

    def __init__(self, optAlgo, learning_rate=0.01, training='sobol', eps=1e-4):
        """"""
        super().__init__()

        self.optimizerDict = {
            'adadelta': optimizers.Adadelta,
            'adagrad': optimizers.Adagrad,
            'adam': optimizers.Adam,
            'adamax': optimizers.Adamax,
            'ftrl': optimizers.Ftrl,
            'nadam': optimizers.Nadam,
            'rmsprop': optimizers.RMSprop,
            'sgd': optimizers.SGD
        }

        self.optimizer = self.optimizerDict[optAlgo](learning_rate)
        self.training = training
        self.eps = eps

    def loadModels(self, stages):
        """"""
        self.numStages = len(stages)
        self.stageGraph = stageGraphDecorator(dgl.DGLGraph())
        self.stageGraph.loadStages(self.stageGraph, stages)
        self.stageGraph.defineConnectivity(self.stageGraph, stages)

    def defineObjectives(self, objectives):
        """"""
        self.numObjectives = len(objectives)
        self.EPOSolver = EPOSolver(self.numObjectives, eps=self.eps)
        self.objectiveFunctions = objectiveFunctions(objectives, self.stageGraph)

    def assignInput(self, input, stageID=None):
        if stageID:
            self.stageGraph.stageInputs[stageID] = input
        else:
            for stageID in self.stageGraph.stageIDs:
                if not tf.is_tensor(input):
                    self.stageGraph.stageInputs[stageID] = tf.convert_to_tensor(input[stageID], dtype=tf.float32)
                else:
                    self.stageGraph.stageInputs[stageID] = input[stageID]


    def assignRandomInput(self):
        """"""
        for stageID in self.stageGraph.stageIDs:
            maxVal = self.stageGraph.stageInputRanges[stageID][:,0]
            bias = self.stageGraph.stageInputRanges[stageID][:,1]
            self.stageGraph.stageInputs[stageID] = tf.random.uniform(shape=[self.stageGraph.stageInputLength[stageID]], minval=-maxVal, maxval=maxVal) + bias
            # self.assignInput(tf.random.uniform(shape=[self.stageGraph.stageInputLength[stageID]], minval=-maxVal, maxval=maxVal) + bias, stageID)

    def getSobolInputs(self, numInputs):
        """"""
        inputs = {}
        inputDictList = []
        for stageID in self.stageGraph.stageIDs:
            maxVal = self.stageGraph.stageInputRanges[stageID][:, 0]
            bias = self.stageGraph.stageInputRanges[stageID][:, 1]
            conditionRanges = [[bias[i]-maxVal[i], bias[i]+maxVal[i]] for i in range(len(maxVal))]
            inputs[stageID] = sobolSequenceConditions(conditionRanges, numInputs, reverse=True)
        for i in range(numInputs):
            inputDict = {}
            for stageID in self.stageGraph.stageIDs:
               inputDict[stageID] = inputs[stageID][i]
            inputDictList.append(inputDict)
        return inputDictList

    def getSobolWeightings(selfself, numObjectiveSamples, numObjectives):
        ranges = [[0,1] for i in range(numObjectives)]
        weightings = sobolSequenceConditions(ranges, numObjectiveSamples, reverse=True)
        return tf.convert_to_tensor(weightings, dtype=tf.float32)

    def calculateModelOutputs(self, nodeDataIdentifier):
        """"""
        self.stageGraph.calculateOutputs(self.stageGraph, nodeDataIdentifier)

    def calculateLinearScalarisationLosses(self, objectiveWeightings, nodeDataIdentifier):
        self.stageGraph.calculateOutputs(self.stageGraph, nodeDataIdentifier)
        objectiveValue = self.objectiveFunctions.calculateWeightedObjective(self.stageGraph, objectiveWeightings,
                                                                            nodeDataIdentifier)
        return objectiveValue

    def calculateIndividualLosses(self, nodeDataIdentifier):
        self.stageGraph.calculateOutputs(self.stageGraph, nodeDataIdentifier)
        lossValues = -self.objectiveFunctions.calculateUnnormalisedLosses(self.stageGraph)
        return lossValues

    def calculateEPOLosses(self, losses, objectiveWeightings, parameters, tape):
        return self.EPOSolver(losses, objectiveWeightings, parameters, tape)

    def epoch(self, trainableVariables):
        """"""
        self.calculateModelOutputs('inputs')
        self.objectiveFunctions.calculateObjectives(self.stageGraph)
        with tf.GradientTape() as tape:
            h_inputs = tf.zeros([0,self.h_dim])
            for stageID in self.stageGraph.stageIDs:
                stuff = tf.reshape(self.stageGraph.stageEncoders[stageID](tf.expand_dims(self.stageGraph.stageInputs[stageID], 0)),[1,self.h_dim])
                h_inputs = tf.concat([h_inputs, stuff], axis=0)
            h_inputs = tf.concat([h_inputs,tf.zeros([self.numObjectives, self.h_dim])], axis=0)
            h = self.multiNet(h_inputs, self.stageGraph)
            for stageID in self.stageGraph.stageIDs:
                unscaledStageGeneratedInputs = tf.squeeze(self.stageGraph.stageDecoders[stageID](
                    tf.expand_dims(h[stageID, :], 0)))
                self.stageGraph.stageGeneratedInputs[stageID] = scaleInputs(unscaledStageGeneratedInputs, self.stageGraph.stageInputRanges[stageID])
            # objectiveValue = self.calculateLinearScalarisationLosses(objectiveWeightings, 'stageGeneratedInputs')
            objectiveValue = self.calculateIndividualLosses('stageGeneratedInputs')
            loss = -objectiveValue
        gradients = tape.gradient(loss, trainableVariables)
        self.optimizer.apply_gradients(zip(gradients, trainableVariables))
        return loss

    def givenInputLoop(self, epochs, trainableVariables):
        for epoch in range(epochs):
            # self.trainableVariablesOld = trainableVariables.copy()
            loss = self.epoch(trainableVariables)
            if epoch % (epochs-1) == 0:
                print('Epoch: {}, Loss: {}'.format(epoch, loss,))
                print('Inputs: ', self.stageGraph.stageInputs.values())
                # print(trainableVariables[0] - self.trainableVariablesOld[0])
                # print('stageOutputs: ',self.stageGraph.stageOutputs.values())
                print('stageGeneratedInputs: ',self.stageGraph.stageGeneratedInputs.values())
                # print('objectiveScalings: ',self.objectiveFunctions.objectiveScalings.values())

    def train(self, numStartSamples, epochs=5, h_dim=5, n_layers=4, network_type='GATConv', **kwargs):
        """"""
        print('Starting training')
        self.stageGraph = dgl.transform.add_self_loop(self.stageGraph)
        trainableVariables = []

        self.h_dim = h_dim

        inputs = self.getSobolInputs(numStartSamples)

        # stage encoders and decoders, used to convert inputs to a standard dimension and outputs from a standard dimension
        for stageID in self.stageGraph.stageIDs:
            self.stageGraph.stageEncoders[stageID] = stageEncoder(self.stageGraph.stageInputLength[stageID],h_dim)
            trainableVariables += self.stageGraph.stageEncoders[stageID].trainable_weights

            self.stageGraph.stageDecoders[stageID] = stageDecoder(self.stageGraph.stageInputLength[stageID], h_dim)
            trainableVariables += self.stageGraph.stageDecoders[stageID].trainable_weights

        self.multiNet = multiNet(self.stageGraph, h_dim, n_layers, trainable=True, network_type=network_type, **kwargs)
        # self.hyperparameterLenDict = createHyperparameterLenDict(self.multiNet)
        trainableVariables += self.multiNet.trainable_weights

        if self.training == 'sobol':
            for inputDict in inputs:
                self.assignInput(inputDict)
                self.givenInputLoop(epochs, trainableVariables)
        else:
            for i in range(numStartSamples):
                self.assignRandomInput()
                self.givenInputLoop(epochs, trainableVariables)

        return self.stageGraph.stageGeneratedInputs.values()

    def getConditions(self):
        self.assignRandomInput()
        self.calculateModelOutputs('inputs')
        self.objectiveFunctions.calculateObjectives(self.stageGraph)
        h_inputs = tf.zeros([0, self.h_dim])
        for stageID in self.stageGraph.stageIDs:
            stuff = tf.reshape(
                self.stageGraph.stageEncoders[stageID](tf.expand_dims(self.stageGraph.stageInputs[stageID], 0)),
                [1, self.h_dim])
            h_inputs = tf.concat([h_inputs, stuff], axis=0)
        h_inputs = tf.concat([h_inputs, tf.zeros([self.numObjectives, self.h_dim])], axis=0)
        h = self.multiNet(h_inputs, self.stageGraph)
        for stageID in self.stageGraph.stageIDs:
            unscaledStageGeneratedInputs = tf.squeeze(self.stageGraph.stageDecoders[stageID](
                tf.expand_dims(h[stageID, :], 0)))
            self.stageGraph.stageGeneratedInputs[stageID] = scaleInputs(unscaledStageGeneratedInputs,
                                                                        self.stageGraph.stageInputRanges[stageID])
        self.calculateModelOutputs('stageGeneratedInputs')
        objectiveValues = objectiveValue = self.calculateIndividualLosses('stageGeneratedInputs')
        return list(self.stageGraph.stageGeneratedInputs.values()), objectiveValues

class objectiveFunctionsPredict():
    """"""

    def __init__(self, objectives, stageGraph):
        self.objectiveIDs = []
        self.objectives = {}
        self.objectveValues = {}
        self.objectiveScalings = {}
        self.objectiveVariableIDs = {}
        self.defineObjectives(objectives, stageGraph)

    def defineObjectives(self, objectives, stageGraph):
        # stageGraph.add_nodes(len(objectives), {'objectives': tf.zeros([len(objectives), 1])})
        num_stages = len(stageGraph.stageIDs)
        for objectiveIndex, objective in enumerate(objectives):
            objectiveID = objectiveIndex + num_stages
            self.objectiveIDs.append(objectiveID)
            self.objectives[objectiveID] = objective.objectiveFunction
            self.objectiveScalings[objectiveID] = -inf
            self.objectiveVariableIDs[objectiveID] = objective.objectiveVariables
            self.objectveValues[objectiveID] = 0
            # if not predict:
            #     stageGraph.add_edges([objectiveID]*len(stageGraph.stageIDs), stageGraph.stageIDs)

    def calculateObjective(self, stageGraph, objectiveID, nodeDataIdentifier='inputs'):
        objectiveVariableValues = []
        for variable in self.objectiveVariableIDs[objectiveID]:
            variableOffset = calculateOffset(stageGraph, variable[0].stageID)
            if variable[1] == 'inputs':
                if nodeDataIdentifier == 'inputs':
                    try:
                        objectiveVariableValues.append(
                            stageGraph.stageInputs[variable[0].stageID][variable[2] - variableOffset])
                    except:
                        objectiveVariableValues.append(stageGraph.stageInputs[variable[0].stageID])
                else:
                    try:
                        objectiveVariableValues.append(
                            stageGraph.stageGeneratedInputs[variable[0].stageID][variable[2] - variableOffset])
                    except:
                        objectiveVariableValues.append(stageGraph.stageGeneratedInputs[variable[0].stageID])
            else:
                try:
                    objectiveVariableValues.append(
                        stageGraph.stageOutputs[variable[0].stageID][variable[2] - variableOffset])
                except:
                    objectiveVariableValues.append(stageGraph.stageOutputs[variable[0].stageID])
        objectiveValue = tf.reshape(self.objectives[objectiveID](*objectiveVariableValues), [1, 1])
        return objectiveValue

    def calculateObjectives(self, stageGraph):
        for ID in self.objectiveIDs:
            objectiveValueUnnorm = self.calculateObjective(stageGraph, ID)
            # stageGraph.nodes[ID].data['objectives'] = self.normaliseObjective(objectiveValueUnnorm, ID)
            self.objectveValues[ID] = objectiveValueUnnorm

    def calculateUnnormalisedLosses(self, stageGraph):
        objectives = self.calculateObjective(stageGraph, self.objectiveIDs[0], 'stageGeneratedInputs')
        if len(self.objectiveIDs) > 1:
            for ID in self.objectiveIDs[1:]:
                obj = self.calculateObjective(stageGraph, ID, 'stageGeneratedInputs')
                objectives = tf.concat([objectives, obj], axis=0)
        return objectives

    def unnormalisedObjectives(self, stageGraph):
        objectiveList = []
        for ID in self.objectiveIDs:
            objectiveValue = self.calculateObjective(stageGraph, ID, 'stageGeneratedInputs')
            objectiveList.append(objectiveValue.numpy())
        return np.squeeze(np.array(objectiveList))

class multi_step_output_prediction():
    """"""

    def __init__(self, optAlgo, learning_rate=0.01, training='sobol', eps=1e-4):
        """"""
        super().__init__()

        self.optimizerDict = {
            'adadelta': optimizers.Adadelta,
            'adagrad': optimizers.Adagrad,
            'adam': optimizers.Adam,
            'adamax': optimizers.Adamax,
            'ftrl': optimizers.Ftrl,
            'nadam': optimizers.Nadam,
            'rmsprop': optimizers.RMSprop,
            'sgd': optimizers.SGD
        }

        self.optimizer = self.optimizerDict[optAlgo](learning_rate)
        self.training = training
        self.eps = eps

    def loadModels(self, stages):
        """"""
        self.numStages = len(stages)
        self.stageGraph = stageGraphDecorator(dgl.DGLGraph())
        self.stageGraph.loadStages(self.stageGraph, stages)
        self.stageGraph.defineConnectivity(self.stageGraph, stages)

    def defineObjectives(self, objectives):
        """"""
        self.numObjectives = len(objectives)
        # self.EPOSolver = EPOSolver(self.numObjectives, eps=self.eps)
        self.objectiveFunctions = objectiveFunctionsPredict(objectives, self.stageGraph)

    def assignInput(self, input, stageID=None):
        if stageID:
            self.stageGraph.stageInputs[stageID] = input
        else:
            for stageID in self.stageGraph.stageIDs:
                if not tf.is_tensor(input):
                    self.stageGraph.stageInputs[stageID] = tf.convert_to_tensor(input[stageID], dtype=tf.float32)
                else:
                    self.stageGraph.stageInputs[stageID] = input[stageID]

    def assignRandomInput(self):
        """"""
        for stageID in self.stageGraph.stageIDs:
            maxVal = self.stageGraph.stageInputRanges[stageID][:, 0]
            bias = self.stageGraph.stageInputRanges[stageID][:, 1]
            self.stageGraph.stageInputs[stageID] = tf.random.uniform(shape=[self.stageGraph.stageInputLength[stageID]],
                                                                     minval=-maxVal, maxval=maxVal) + bias
            # self.assignInput(tf.random.uniform(shape=[self.stageGraph.stageInputLength[stageID]], minval=-maxVal, maxval=maxVal) + bias, stageID)

    def getSobolInputs(self, numInputs):
        """"""
        inputs = {}
        inputDictList = []
        for stageID in self.stageGraph.stageIDs:
            maxVal = self.stageGraph.stageInputRanges[stageID][:, 0]
            bias = self.stageGraph.stageInputRanges[stageID][:, 1]
            conditionRanges = [[bias[i] - maxVal[i], bias[i] + maxVal[i]] for i in range(len(maxVal))]
            inputs[stageID] = sobolSequenceConditions(conditionRanges, numInputs, reverse=True)
        for i in range(numInputs):
            inputDict = {}
            for stageID in self.stageGraph.stageIDs:
                inputDict[stageID] = inputs[stageID][i]
            inputDictList.append(inputDict)
        return inputDictList

    def getSobolWeightings(selfself, numObjectiveSamples, numObjectives):
        ranges = [[0, 1] for i in range(numObjectives)]
        weightings = sobolSequenceConditions(ranges, numObjectiveSamples, reverse=True)
        return tf.convert_to_tensor(weightings, dtype=tf.float32)

    def calculateModelOutputs(self, nodeDataIdentifier):
        """"""
        self.stageGraph.calculateOutputs(self.stageGraph, nodeDataIdentifier)

        # return outputs

    def loss(self, objectiveValues, objectivePreds):
        """"""
        # print('objectiveValues: ',objectiveValues)
        # print('objectivePreds: ',objectivePreds)
        # loss = tf.keras.losses.KLD(objectiveValues, objectivePreds)
        loss = tf.keras.losses.mean_squared_error(objectiveValues,objectivePreds)
        # print(loss)
        return tf.reduce_sum(loss)

    def calculateActualObjectives(self, inputs):
        self.assignInput(inputs[0])
        self.calculateModelOutputs('inputs')
        self.objectiveFunctions.calculateObjectives(self.stageGraph)
        # print(self.objectiveFunctions.objectveValues)
        objectiveValues = [list(self.objectiveFunctions.objectveValues.values())]
        for input in inputs[1:]:
            self.assignInput(input)
            self.calculateModelOutputs('inputs')
            self.objectiveFunctions.calculateObjectives(self.stageGraph)
            objectiveValues.append(list(self.objectiveFunctions.objectveValues.values()))
        return np.squeeze(np.array(objectiveValues))

    def givenInputLoop(self):
        # self.calculateModelOutputs('inputs')
        # self.objectiveFunctions.calculateObjectives(self.stageGraph)
        # objectiveValues = self.objectiveFunctions.objectveValues
        h_inputs = tf.zeros([0, self.h_dim])
        for stageID in self.stageGraph.stageIDs:
            stuff = tf.reshape(
                self.stageGraph.stageEncoders[stageID](tf.expand_dims(self.stageGraph.stageInputs[stageID], 0)),
                [1, self.h_dim])
            h_inputs = tf.concat([h_inputs, stuff], axis=0)
        h = self.multiNet(h_inputs, self.stageGraph)
        h = tf.reshape(h, [1,-1])
        objectivePreds = self.stageGraph.objectiveDecoder(h)
        return objectivePreds #objectiveValues,

    def epoch(self, inputs, trainableVariables, objectiveValues):
        """"""
        with tf.GradientTape() as tape:
            self.assignInput(inputs[0])
            objectivePreds = self.givenInputLoop() #objectiveValue,
            # print(objectiveValue, objectivePred)
            # objectiveValues = tf.convert_to_tensor(list(objectiveValue.values()))
            # objectivePreds = objectivePred
            # print('objectivePreds: ',objectivePreds)
            for inputDict in inputs[1:]:
                self.assignInput(inputDict)
                objectivePred = self.givenInputLoop()#objectiveValue,
                # print('objectivePred: ', objectivePred)
                # objectiveValues = tf.concat([objectiveValues,tf.convert_to_tensor(list(objectiveValue.values()))], axis=1)
                objectivePreds = tf.concat([objectivePreds, objectivePred], axis=1)
            objectivePreds = tf.squeeze(objectivePreds)
            # print('objectivePreds: ',objectivePreds)
            loss = self.loss(objectiveValues, objectivePreds)
        gradients = tape.gradient(loss, trainableVariables)
        # print('gradients: ',gradients)
        self.optimizer.apply_gradients(zip(gradients, trainableVariables))
        return loss

    def train(self, numStartSamples, epochs=5, h_dim=5, n_layers=4, network_type='GATConv', **kwargs):
        """"""
        print('Starting training')
        self.stageGraph = dgl.transform.add_self_loop(self.stageGraph)
        trainableVariables = []

        self.h_dim = h_dim

        inputs = self.getSobolInputs(numStartSamples)
        # print('inputs: ',inputs)

        # stage encoders and decoders, used to convert inputs to a standard dimension and outputs from a standard dimension
        for stageID in self.stageGraph.stageIDs:
            self.stageGraph.stageEncoders[stageID] = stageEncoder(self.stageGraph.stageInputLength[stageID], h_dim)
            trainableVariables += self.stageGraph.stageEncoders[stageID].trainable_weights

            # self.stageGraph.stageDecoders[stageID] = stageDecoder(self.stageGraph.stageInputLength[stageID], h_dim)
        self.stageGraph.objectiveDecoder = objectiveDecoder(self.numObjectives, self.numStages, h_dim)
        trainableVariables += self.stageGraph.objectiveDecoder.trainable_weights

        self.multiNet = multiNetPredict(self.stageGraph, h_dim, n_layers, trainable=True, network_type=network_type, **kwargs)
        trainableVariables += self.multiNet.trainable_weights

        actualObjectiveValues = self.calculateActualObjectives(inputs)
        actualMean = np.mean(actualObjectiveValues)
        # actualObjectiveValues -= actualMean
        print(actualMean)
        print(actualObjectiveValues)

        # print('actual: ', actualObjectiveValues)
        losses = []
        for epoch in range(epochs):
            loss = self.epoch(inputs, trainableVariables, actualObjectiveValues)
            losses.append(loss)
            print('Epoch: {}, Loss: {}'.format(epoch, loss))
            # if epoch % 40 == 0:
            #     print(trainableVariables)

        return actualMean, losses

    def get_results(self, numStartSamples, mean):
        # inputs = self.getSobolInputs(numStartSamples)
        # actualObjectiveValues = self.calculateActualObjectives(inputs)
        # self.assignInput(inputs[0])
        self.assignRandomInput()
        self.calculateModelOutputs('inputs')
        self.objectiveFunctions.calculateObjectives(self.stageGraph)
        actualObjectiveValues = [list(self.objectiveFunctions.objectveValues.values())]
        objectivePred = self.givenInputLoop() #objectiveValue,
        # objectiveValues = tf.convert_to_tensor(list(objectiveValue.values()))
        objectivePreds = objectivePred
        for _ in range(numStartSamples-1):
            # self.assignInput(inputDict)
            self.assignRandomInput()
            self.calculateModelOutputs('inputs')
            self.objectiveFunctions.calculateObjectives(self.stageGraph)
            actualObjectiveValues.append(list(self.objectiveFunctions.objectveValues.values()))
            objectivePred = self.givenInputLoop()#objectiveValue,
            # objectiveValues = tf.concat([objectiveValues, tf.convert_to_tensor(list(objectiveValue.values()))], axis=1)
            objectivePreds = tf.concat([objectivePreds, objectivePred], axis=1)
        # objectivePreds += mean
        return np.squeeze(np.array(actualObjectiveValues)), objectivePreds.numpy()