import tensorflow as tf
from models import *
from utils import *
import dgl
import tensorflow.keras.optimizers as optimizers
import numpy as np
from random import uniform

def stageGraphDecorator(graph):
    setattr(graph, 'stages', {})
    setattr(graph, 'stageIDs', [])
    setattr(graph, 'stageInputs', {})
    setattr(graph, 'stageOutputs', {})
    setattr(graph, 'stageGeneratedInputs', {})
    setattr(graph, 'stagePrevious', {})
    setattr(graph, 'stageParameters', {})
    setattr(graph, 'stageInputRanges', {})
    setattr(graph, 'stageInputLength', {})
    setattr(graph, 'stageOutputLength', {})
    setattr(graph, 'stageOutput', {})
    setattr(graph, 'stageEncoders', {})
    setattr(graph, 'stageDecoders', {})

    def loadStages(self, stages):
        for stage in stages:
            self.stageIDs.append(stage.stageID)
            self.stageParameters[stage.stageID] = stage.model.trainable_variables
            self.stages[stage.stageID] = stage.model
            self.stageInputRanges[stage.stageID] = stage.inputRanges
            self.stageInputLength[stage.stageID] = stage.model.x_dim
            # print(self.stageInputLength[stage.stageID])
            self.stageOutputLength[stage.stageID] = stage.model.y_dim
            # print(tf.zeros([1, self.stageInputLength[stage.stageID]]))
            # self.add_nodes(1, {'inputs': tf.zeros([1, self.stageInputLength[stage.stageID]])})
            # self.stageInput[stage.stageID] = []
            # print(self.nodes[stage.stageID].data['inputs'])

    setattr(graph, 'loadStages', loadStages)

    def defineConnectivity(self, stages):
        for stage in stages:
            if not stage.followingStages == None:
                for followingStage in stage.followingStages:
                    self.stageInputLength[followingStage[0]] -= len(followingStage[1])
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
                    for stageVar in previousStage[1]:
                        previousStageOutputs.append(self.stageOutputs[previousStage[0]][stageVar])
                modelInputs = tf.concat([previousStageOutputs, self.stageInputs[stageID]],0)
            except:
                modelInputs = self.stageInputs[stageID]
            if nodeDataIdentifier == 'inputs':
                output = self.stages[stageID].evaluateModel(modelInputs)
            elif nodeDataIdentifier == 'stageGeneratedInputs':
                output = self.stages[stageID].evaluateModel(modelInputs)
            else:
                output = self.stages[stageID].evaluateModel(modelInputs)
            self.stageOutputs[stageID] = output

    setattr(graph, 'calculateOutputs', calculateOutputs)

    def applyInputs(self, inputs):
        endIndex = 0
        for stageID in self.stageIDs:
            startIndex = endIndex
            endIndex = endIndex + self.stageInputLength[stageID]
            # self.nodes[stageID].data['inputs'] = inputs[startIndex:endIndex]
            self.stageInputs[stageID] = inputs[startIndex:endIndex]

    setattr(graph, 'applyInputs', applyInputs)

    return graph

class objectiveFunctions():
    """"""

    def __init__(self, objectives, stageGraph):
        self.objectiveIDs = []
        self.objectives = {}
        self.objectiveVariableIDs = {}
        self.defineObjectives(objectives, stageGraph)

    def defineObjectives(self, objectives, stageGraph):
        stageGraph.add_nodes(len(objectives), {'objectives': tf.zeros([len(objectives), 1])})
        num_stages = len(stageGraph.stageIDs)
        for objectiveIndex, objective in enumerate(objectives):
            objectiveID = objectiveIndex + num_stages
            self.objectiveIDs.append(objectiveID)
            self.objectives[objectiveID] = objective.objectiveFunction
            self.objectiveVariableIDs[objectiveID] = objective.objectiveVariables
            stageGraph.add_edges([objectiveID]*len(stageGraph.stageIDs), stageGraph.stageIDs)

    def calculateObjectives(self, stageGraph):
        for index, ID in enumerate(self.objectiveIDs):
            objectiveVariableValues = []
            for variable in self.objectiveVariableIDs[ID]:
                if variable[1] == 'inputs':
                    objectiveVariableValues.append(stageGraph.stageInputs[variable[0].stageID][variable[2]])
                else:
                    objectiveVariableValues.append(stageGraph.stageOutputs[variable[0].stageID][variable[2]])
            stageGraph.nodes[ID].data['objectives'] = tf.reshape(self.objectives[ID](*objectiveVariableValues),[1,1])

    def calculateWeightedObjective(self, stageGraph, objectiveWeightings):
        """"""
        objectiveValue = 0
        for index, ID in enumerate(self.objectiveIDs):
            objectiveVariableValues = []
            for variable in self.objectiveVariableIDs[ID]:
                # objectiveVariableValues.append(stageGraph.nodes[variable[0].stageID].data[variable[1]][variable[2]])
                if variable[1] == 'inputs':
                    objectiveVariableValues.append(stageGraph.stageInputs[variable[0].stageID][variable[2]])
                else:
                    objectiveVariableValues.append(stageGraph.stageOutputs[variable[0].stageID][variable[2]])
            objectiveValue += self.objectives[ID](*objectiveVariableValues) * objectiveWeightings[0, index]
        return objectiveValue

class multi_step():
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
        self.stageGraph = stageGraphDecorator(dgl.DGLGraph())
        self.stageGraph.loadStages(self.stageGraph, stages)
        self.stageGraph.defineConnectivity(self.stageGraph, stages)

    def defineObjectives(self, objectives):
        """"""
        self.objectiveFunctions = objectiveFunctions(objectives, self.stageGraph)

    def assignRandomInput(self):
        """"""
        for stageID in self.stageGraph.stageIDs:
            self.stageGraph.stageInputs[stageID] = tf.convert_to_tensor([uniform(*range) for range in self.stageGraph.stageInputRanges[stageID]])

    def calculateModelOutputs(self, objectiveWeightings, nodeDataIdentifier):
        """"""
        self.stageGraph.calculateOutputs(self.stageGraph, nodeDataIdentifier)
        objectiveValue = self.objectiveFunctions.calculateWeightedObjective(self.stageGraph, objectiveWeightings)
        return objectiveValue

    def epoch(self, objectiveWeightings, trainableVariables):
        """"""
        self.assignRandomInput()
        self.calculateModelOutputs(objectiveWeightings, 'inputs')
        self.objectiveFunctions.calculateObjectives(self.stageGraph)
        with tf.GradientTape() as tape:
            hyperparametersDict = self.multiHyperNet(objectiveWeightings)
            updateHyperparameters(hyperparametersDict, self.multiNet)
            # print(self.stageGraph.ndata['objectives'])
            # print(trainableVariables)
            for stageID in self.stageGraph.stageIDs:
                self.stageGraph.nodes[stageID].data['h_inputs'] = self.stageGraph.stageEncoders[stageID](
                    tf.expand_dims(self.stageGraph.stageInputs[stageID], 0))
            h = self.multiNet(self.stageGraph)
            for stageID in self.stageGraph.stageIDs:
                self.stageGraph.stageGeneratedInputs[stageID] = self.stageGraph.stageDecoders[stageID](
                    tf.expand_dims(h[stageID, :], 0))
            print(self.stageGraph.stageGeneratedInputs.values())
            objectiveValue = self.calculateModelOutputs(objectiveWeightings, 'stageGeneratedInputs')
            loss = -objectiveValue
            print('loss: ', loss)
        gradients = tape.gradient(loss, trainableVariables)
        # gradients = tape.gradient(loss, self.multiHyperNet.trainable_weights)
        print(gradients)
        self.optimizer.apply_gradients(zip(gradients, trainableVariables))
        return loss

    def optimise(self, numObjectives, preferenceEncodeDim, hyperparameterGeneratorDim, epochs=5, h_dim=5, n_layers=4, network_type='GATConv', **kwargs):
        """"""
        self.stageGraph = dgl.transform.add_self_loop(self.stageGraph)
        objectiveWeightings = np.expand_dims(np.array([0.5, 0.5]), axis=0)
        # objectiveWeightings = np.array([0.5, 0.5])
        trainableVariables = []

        # stage encoders and decoders, used to convert inputs to a standard dimension and outputs from a standard dimension
        for stageID in self.stageGraph.stageIDs:
            self.stageGraph.stageEncoders[stageID] = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(1, self.stageGraph.stageInputLength[stageID])),
                tf.keras.layers.Dense(h_dim)
            ])
            trainableVariables = trainableVariables + self.stageGraph.stageEncoders[stageID].trainable_weights

            self.stageGraph.stageDecoders[stageID] = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(1, h_dim + 1)),
                tf.keras.layers.Dense(self.stageGraph.stageInputLength[stageID])
            ])
            trainableVariables = trainableVariables + self.stageGraph.stageDecoders[stageID].trainable_weights

        self.multiNet = multiNet(self.stageGraph, h_dim, n_layers, network_type, **kwargs)
        hyperparameterLenDict = createHyperparameterLenDict(self.multiNet)

        self.multiHyperNet = multiHyperNet(numObjectives, preferenceEncodeDim, hyperparameterGeneratorDim, hyperparameterLenDict)
        # print(self.multiHyperNet.trainable_weights)
        trainableVariables = trainableVariables + self.multiHyperNet.trainable_weights

        for epoch in range(epochs):
            loss = self.epoch(objectiveWeightings, trainableVariables)
            if epoch % 1 == 0:
                print('Epoch: ', epoch,', Loss: ',loss)