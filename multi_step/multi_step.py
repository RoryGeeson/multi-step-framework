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
                output = tf.squeeze(self.stages[stageID].predict_f(Xnew=tf.expand_dims(modelInputs,0))[0])
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
        # if objectiveValue > self.objectiveScalings[objectiveID]:
        #     self.objectiveScalings[objectiveID] = objectiveValue
        # return objectiveValue / tf.math.abs(self.objectiveScalings[objectiveID])
        # if objectiveValue < 0:
        #     return self.objectiveScalings[objectiveID] / tf.math.abs(objectiveValue)
        # else:
        #     return objectiveValue / tf.math.abs(self.objectiveScalings[objectiveID])


        # if objectiveValue < 0:
        #     if objectiveValue > self.objectiveScalings[objectiveID]:
        #         self.objectiveScalings[objectiveID] = objectiveValue
        #     return objectiveValue / tf.math.abs(self.objectiveScalings[objectiveID])
        # else:
        #     if self.objectiveScalings[objectiveID]

        # if tf.math.abs(objectiveValue) > self.objectiveScalings[objectiveID]:
        #     self.objectiveScalings[objectiveID] = tf.math.abs(objectiveValue)
        # return objectiveValue / self.objectiveScalings[objectiveID]

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
        # print(stageGraph.stageInputs.values())
        # print('objectiveVariableValues: ',objectiveVariableValues)
        objectiveValue = tf.reshape(self.objectives[objectiveID](*objectiveVariableValues), [1, 1])
        # if objectiveValue > self.objectiveScalings[objectiveID]:
        #     self.objectiveScalings[objectiveID] = objectiveValue
        # return objectiveValue / self.objectiveScalings[objectiveID]
        return objectiveValue

    def calculateObjectives(self, stageGraph):
        for ID in self.objectiveIDs:
            # stageGraph.nodes[ID].data['objectives'] = self.calculateObjective(stageGraph, ID)
            objectiveValueUnnorm = self.calculateObjective(stageGraph, ID)
            stageGraph.nodes[ID].data['objectives'] = self.normaliseObjective(objectiveValueUnnorm, ID)

    def calculateWeightedObjective(self, stageGraph, objectiveWeightings, nodeDataIdentifier):
        """"""
        objectiveValue = 0
        for index, ID in enumerate(self.objectiveIDs):
            objectiveValueUnnorm = self.calculateObjective(stageGraph, ID, nodeDataIdentifier)
            objectiveValue += self.normaliseObjective(objectiveValueUnnorm,ID) * objectiveWeightings[0, index]
            # objectiveValue += self.calculateObjective(stageGraph, ID, nodeDataIdentifier) * objectiveWeightings[0, index]
        return objectiveValue

    def unnormalisedObjectives(self, stageGraph):
        objectiveList = []
        for ID in self.objectiveIDs:
            objectiveValue = self.calculateObjective(stageGraph, ID, 'stageGeneratedInputs')
            objectiveList.append((objectiveValue.numpy()))
        return np.squeeze(np.array(objectiveList))

    def resetScalings(self):
        for scalingKey in self.objectiveScalings.keys():
            self.objectiveScalings[scalingKey] = -inf

class multi_step_graph():
    """"""

    def __init__(self, optAlgo, learning_rate=0.01):
        """"""
        super().__init__()

        self.optimizerDict = {
            'adadelta': optimizers.Adadelta(learning_rate),
            'adagrad': optimizers.Adagrad(learning_rate),
            'adam': optimizers.Adam(learning_rate),
            'adamax': optimizers.Adamax(learning_rate),
            'ftrl': optimizers.Ftrl(learning_rate),
            'nadam': optimizers.Nadam(learning_rate),
            'rmsprop': optimizers.RMSprop(learning_rate),
            'sgd': optimizers.SGD(learning_rate)
        }

        self.optimizer = self.optimizerDict[optAlgo]

    def loadModels(self, stages):
        """"""
        self.numStages = len(stages)
        self.stageGraph = stageGraphDecorator(dgl.DGLGraph())
        self.stageGraph.loadStages(self.stageGraph, stages)
        self.stageGraph.defineConnectivity(self.stageGraph, stages)

    def defineObjectives(self, objectives):
        """"""
        self.numObjectives = len(objectives)
        self.objectiveFunctions = objectiveFunctions(objectives, self.stageGraph)

    def assignRandomInput(self):
        """"""
        for stageID in self.stageGraph.stageIDs:
            maxVal = self.stageGraph.stageInputRanges[stageID][:,0]
            bias = self.stageGraph.stageInputRanges[stageID][:,1]
            self.stageGraph.stageInputs[stageID] = tf.random.uniform(shape=[self.stageGraph.stageInputLength[stageID]], minval=-maxVal, maxval=maxVal) + bias

    def calculateModelOutputs(self, nodeDataIdentifier):
        """"""
        self.stageGraph.calculateOutputs(self.stageGraph, nodeDataIdentifier)

    def calculateObjectiveValue(self, objectiveWeightings, nodeDataIdentifier):
        self.stageGraph.calculateOutputs(self.stageGraph, nodeDataIdentifier)
        objectiveValue = self.objectiveFunctions.calculateWeightedObjective(self.stageGraph, objectiveWeightings,
                                                                            nodeDataIdentifier)
        return objectiveValue

    def epoch(self, objectiveWeightings, trainableVariables):
        """"""
        self.assignRandomInput()
        # print('Random Inputs: ',self.stageGraph.stageInputs.values())
        self.calculateModelOutputs('inputs')
        # print('Caluclated Outputs: ',self.stageGraph.stageOutputs.values())
        self.objectiveFunctions.calculateObjectives(self.stageGraph)
        with tf.GradientTape() as tape:
            hyperparameters = self.multiHyperNet(objectiveWeightings)
            # print(hyperparameters)
            updateHyperparameters(self.hyperparameterLenDict, hyperparameters, self.multiNet)
            h_inputs = tf.zeros([0,self.h_dim])
            for stageID in self.stageGraph.stageIDs:
                stuff = tf.reshape(self.stageGraph.stageEncoders[stageID](tf.expand_dims(self.stageGraph.stageInputs[stageID], 0)),[1,self.h_dim])
                h_inputs = tf.concat([h_inputs, stuff], axis=0)
                # h_inputs = tf.concat([h_inputs, self.stageGraph.stageEncoders[stageID](tf.expand_dims(self.stageGraph.stageInputs[stageID], 0))], axis=0)
            h_inputs = tf.concat([h_inputs,tf.zeros([self.numObjectives, self.h_dim])], axis=0)
            h = self.multiNet(h_inputs, self.stageGraph)
            for stageID in self.stageGraph.stageIDs:
                unscaledStageGeneratedInputs = tf.squeeze(self.stageGraph.stageDecoders[stageID](
                    tf.expand_dims(h[stageID, :], 0)))
                self.stageGraph.stageGeneratedInputs[stageID] = scaleInputs(unscaledStageGeneratedInputs, self.stageGraph.stageInputRanges[stageID])
            objectiveValue = self.calculateObjectiveValue(objectiveWeightings, 'stageGeneratedInputs')
            print('ObjectiveValue: ',objectiveValue)
            loss = -objectiveValue
        gradients = tape.gradient(loss, trainableVariables)
        self.optimizer.apply_gradients(zip(gradients, trainableVariables))
        return loss

    def train(self, numObjectiveSamples, preferenceEncodeDim, hyperparameterGeneratorDim, epochs=5, h_dim=5, n_layers=4, network_type='GATConv', **kwargs):
        """"""
        print('Starting training')
        self.stageGraph = dgl.transform.add_self_loop(self.stageGraph)
        trainableVariables = []

        self.h_dim = h_dim

        # stage encoders and decoders, used to convert inputs to a standard dimension and outputs from a standard dimension
        for stageID in self.stageGraph.stageIDs:
            self.stageGraph.stageEncoders[stageID] = stageEncoder(self.stageGraph.stageInputLength[stageID],h_dim)
            trainableVariables += self.stageGraph.stageEncoders[stageID].trainable_weights

            self.stageGraph.stageDecoders[stageID] = stageDecoder(self.stageGraph.stageInputLength[stageID], h_dim)
            trainableVariables += self.stageGraph.stageDecoders[stageID].trainable_weights

        self.multiNet = multiNet(self.stageGraph, h_dim, n_layers, network_type, **kwargs)
        self.hyperparameterLenDict = createHyperparameterLenDict(self.multiNet)

        self.multiHyperNet = multiHyperNet(self.numObjectives, preferenceEncodeDim, hyperparameterGeneratorDim, self.hyperparameterLenDict)
        trainableVariables = trainableVariables + self.multiHyperNet.trainable_weights

        for objectiveWeightings in tf.linalg.normalize(tf.random.uniform(shape=[numObjectiveSamples,self.numObjectives], maxval=1), ord=1, axis=1)[0]:
            self.objectiveFunctions.resetScalings()
            for epoch in range(epochs):
                loss = self.epoch(tf.expand_dims(objectiveWeightings, 0), trainableVariables)
                if epoch % 1 == 0:
                    print('Weightings: {}, Epoch: {}, Loss: {}, Objective values: {}'.format(objectiveWeightings.numpy(),epoch,loss, self.objectiveFunctions.unnormalisedObjectives(self.stageGraph)))
                    # print('stageOutputs: ',self.stageGraph.stageOutputs.values())
                    print('stageGeneratedInputs: ',self.stageGraph.stageGeneratedInputs.values())
                    print('objectiveScalings: ',self.objectiveFunctions.objectiveScalings.values())

        return self.stageGraph.stageGeneratedInputs.values()

    def getConditions(self, objectiveWeightings):
        objectiveWeightings = tf.convert_to_tensor([objectiveWeightings])
        self.assignRandomInput()
        self.calculateModelOutputs('inputs')
        self.objectiveFunctions.calculateObjectives(self.stageGraph)
        hyperparameters = self.multiHyperNet(objectiveWeightings)
        # print(hyperparameters)
        updateHyperparameters(self.hyperparameterLenDict, hyperparameters, self.multiNet)
        h_inputs = tf.zeros([0, self.h_dim])
        for stageID in self.stageGraph.stageIDs:
            stuff = tf.reshape(
                self.stageGraph.stageEncoders[stageID](tf.expand_dims(self.stageGraph.stageInputs[stageID], 0)),
                [1, self.h_dim])
            h_inputs = tf.concat([h_inputs, stuff], axis=0)
            # h_inputs = tf.concat([h_inputs, self.stageGraph.stageEncoders[stageID](
            #     tf.expand_dims(self.stageGraph.stageInputs[stageID], 0))], axis=0)
        h_inputs = tf.concat([h_inputs, tf.zeros([self.numObjectives, self.h_dim])], axis=0)
        h = self.multiNet(h_inputs, self.stageGraph)
        for stageID in self.stageGraph.stageIDs:
            unscaledStageGeneratedInputs = tf.squeeze(self.stageGraph.stageDecoders[stageID](
                tf.expand_dims(h[stageID, :], 0)))
            self.stageGraph.stageGeneratedInputs[stageID] = scaleInputs(unscaledStageGeneratedInputs,
                                                                        self.stageGraph.stageInputRanges[stageID])
        self.calculateModelOutputs('stageGeneratedInputs')
        objectiveValues = self.objectiveFunctions.unnormalisedObjectives(self.stageGraph)
        return list(self.stageGraph.stageGeneratedInputs.values()), objectiveValues