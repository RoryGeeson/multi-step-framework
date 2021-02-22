from math import exp
import numpy as np
from scipy.integrate import odeint
from itertools import product
import sobol_seq as sob
from neural_process import *

def relu(x):
    return max(0,x)

def bound_upper(x):
    return min(1,x)

class Toy():
    """"""

    def __init__(self):

        # self.noiseMaker = keras.Sequential([
        #     keras.layers.GaussianNoise(0.1)
        # ])

        self.splitDict = {1:0.2,2:0.3,3:0.4,4:0.6,5:0.7}

        self.yieldWeightings = [0,0,1,1,3,2,3]

        self.finalProductPower = 1.5

        self.splitCostDict = {1: 1, 2: 3, 3: 4, 4: 6, 5: 7}

        self.temperatureCostConstants = [3,1.1]

        self.pressureCostConstants = [2, 2.3]

        self.timeCostConstants = [1, 1.2]

        self.conditionTypes = ['Temp', 'Pressure', 'time', 'Temp', 'Pressure', 'time', 'Temp', 'Pressure', 'time']

    def stage_1_ODEs(self, species, t, T1, P1):
        """
        Forms an ODE that defines the reaction taking place in the first stage, A -> B + C
        :param t:
        :param T1:
        :param P1:
        :param solvent:
        :return:
        """

        # Define Arrhenius pre-exponential factor
        A_B = 5
        A_C = 10

        # Define activation energy
        E_B = 5
        E_C = 10

        # Define rate equations
        dxB_dt = A_B * exp(- E_B / T1) * P1 * species[0]
        dxC_dt = A_C * exp(- E_C / T1) / P1 * species[0]
        return -dxB_dt-dxC_dt, dxB_dt, dxC_dt

    def stage_2_ODEs(self, species, t, T2, P2):
        """
        Forms an ODE that defines the reaction taking place in the second stage, A + B -> D + E
        :param t:
        :param T2:
        :param P2:
        :return:
        """

        # Define Arrhenius pre-exponential factor
        A_D = 5
        A_E = 10

        # Define activation energy
        E_D = 5
        E_E = 10

        # Define rate equations
        dxD_dt = A_D * exp(- E_D / T2) * P2 * species[0] * species[1]
        dxE_dt = A_E * exp(- E_E / T2) / P2 * species[0] * species[1]
        return -dxD_dt, -dxD_dt, dxD_dt, dxE_dt

    def stage_3_ODEs(self, species, t, T3, P3):
        """
        Forms an ODE that defines the reaction taking place in the third stage, C + D + E -> F
        :param t:
        :param T3:
        :param P3:
        :return:
        """

        # Define Arrhenius pre-exponential factor
        A_F = 5

        # Define activation energy
        E_F = 5

        # Define rate equations
        dxF_dt = A_F * exp(- E_F / T3) * P3 * species[0] * species[1] * species[2]
        return -dxF_dt, -dxF_dt, -dxF_dt, dxF_dt

    def noiseMaker(self, x, noise_scale=0.025):
        return np.random.normal(x,noise_scale*x)

    def stage_0(self, A, split_discrete=3):
        A1 = A * self.splitDict[round(split_discrete)]
        A2 = A - A1
        return A1,A2

    def stage_1(self, A1, T1, P1, t_out):
        t_out = np.linspace(0, t_out, 500)
        species = [1, 0, 0] # x_A1, x_B, x_C
        species_out = odeint(self.stage_1_ODEs, species, t_out, (T1, P1))
        B = species_out[-1][1] * A1
        C = species_out[-1][2] * A1
        A1 = B - C
        return A1, B, C

    def stage_2(self, A2, B, T2, P2, t_out):
        t_out = np.linspace(0, t_out, 500)
        total = A2 + B
        x_A2 = A2 / total
        x_B = 1-x_A2
        species = [x_A2, x_B, 0, 0] # x_A2, x_B, x_D, x_E
        species_out = odeint(self.stage_2_ODEs, species, t_out, (T2, P2))
        D = species_out[-1][2] * total
        E = species_out[-1][3] * total
        A2 = species_out[-1][0] * total
        B = total - A2 - D - E
        return A2, B, D, E

    def stage_3(self, C, D, E, T3, P3, t_out):
        t_out = np.linspace(0, t_out, 500)
        total = C + D + E
        x_C = C / total
        x_D = D / total
        species = [x_C, x_D, 1-x_C-x_D, 0]
        species_out = odeint(self.stage_3_ODEs, species, t_out, (T3, P3))
        F = species_out[-1][3] * total
        C = species_out[-1][0] * total
        D = species_out[-1][1] * total
        E = total - C- D - F
        return C, D, E, F

    def noiselessExperimentalRun(self, A, conditions):
        yieldList = []
        A1, A2 = self.stage_0(A)
        yieldList += [A1, A2]
        A1, B, C = self.stage_1(A1, *conditions[0:3])
        yieldList += [A1, B, C]
        A2, B, D, E = self.stage_2(A2, B, *conditions[3:6])
        yieldList += [A2, B, D, E]
        C, D, E, F = self.stage_3(C, D, E, *conditions[6:])
        yieldList += [C, D, E, F]
        # return [A1, A2, A1, B, C, A2, B, D, E, C, D, E, F]
        return np.array(yieldList)

    def experimentalRun(self, A, conditions):
        """
        Generates yields of an experimental run at given conditions, adding normally distributed noise to the results
        :param conditions: list of the experimental conditions [split, T1, P1, solvent, t_out1, T2, P2, t_out2, T3, P3, t_out3]
        :return: list of experimental yields [A1, A2, B, C, D, E, F]
        """
        noiselessData = np.array(self.noiselessExperimentalRun(A, conditions)).clip(0)
        experimentData = self.noiseMaker(noiselessData)
        experimentData = [relu(data) for data in experimentData]
        costs = self.costs(conditions)
        return experimentData, costs

    def yieldObjective(self, yieldList):
        yieldList[-1] = yieldList[-1]**self.finalProductPower
        objective = np.multiply(yieldPower,self.yieldWeightings)
        return objective

    def solventCost(self, solvent):
        return self.solventCostDict[round(solvent)]

    def splitCost(self, split_discrete):
        return self.splitCostDict[round(split_discrete)]

    def temperatureCost(self, temp):
        tempCost = self.temperatureCostConstants[0] * temp ** self.temperatureCostConstants[1]
        return tempCost

    def pressureCost(self, pressure):
        pressureCost = self.pressureCostConstants[0] * pressure ** self.pressureCostConstants[1]
        return pressureCost

    def timeCost(self, time):
        timeCost = self.timeCostConstants[0] * time ** self.timeCostConstants[1]
        return timeCost

    # ['Temp', 'Pressure', 'time', 'Temp', 'Pressure', 'time', 'Temp', 'Pressure', 'time']

    def costs(self, conditions):
        try:
            condIter = np.ndenumerate(conditions)
        except:
            condIter = enumerate(conditions)
        costs = []
        for ind, condition in condIter:
            conditionType = self.conditionTypes[int(ind[0])]
            if conditionType == 'split':
                costs.append(self.splitCost(condition))
            elif conditionType == 'Temp':
                costs.append((self.temperatureCost(condition)))
            elif conditionType == 'Pressure':
                costs.append(self.pressureCost(condition))
            elif conditionType == 'solvent':
                costs.append(self.solventCost(condition))
            else:
                costs.append(self.timeCost(condition))
        return costs
    #
    # def calculateObjectivelist(self, ):
    #     return
    #
    # def addDataLine(self, conditions):
    #     objectives = self.calculateObjectivelist()
    #     try:
    #         self.data.append()
    #     else:
    #         self.data =

    # def randomConditions(self, tempRange, pressureRange, solventRange, splitRange):
    #
    #     return []

    def allConditions(self, condition_ranges, continuous_num=5):
        conditions = []
        for num, condition in enumerate(condition_ranges):
            try:
                conditions.append(np.arange(condition[0],condition[1],condition[2]))
            except:
                conditions.append(np.linspace(condition[0], condition[1], num=continuous_num))
        conditions = product(*conditions)
        return list(conditions)


    def sobolSequenceConditions(self, condition_ranges, num_points):
        """"""
        dimensions = len(condition_ranges)
        sobol_sequence = sob.i4_sobol_generate(dimensions, num_points)
        lower_bounds = np.array([condition[0] for condition in condition_ranges])
        upper_bounds = np.array([condition[1] for condition in condition_ranges])
        ranges = upper_bounds - lower_bounds
        offset = np.tile(lower_bounds,(num_points,1))
        conditions = sobol_sequence * ranges + offset
        return np.array(conditions)

    def fit_stage_1(self, data, epochs=1000):
        # x_dim, y_dim, layer_dim, r_dim, z_dim, num_samples, optimizer, learning_rate
        model_stage_1 = NeuralProcess(3, 3, 12, 12, 6, 5, 'adam', 0.001)
        for epoch in range(epochs):
            # data, target_fraction=0.2
            # [x, y]; x: function input (n_points,len_x), y: function output (n_points, len_y)
            loss = model_stage_1.epoch(data)
            if epoch % 100 == 0:
                print('epoch: ',epoch,', loss = ',loss.numpy())
        self.stage_1_model = model_stage_1

    def fit_stage_2(self, data, epochs=1000):
        # x_dim, y_dim, layer_dim, r_dim, z_dim, num_samples, optimizer, learning_rate
        model_stage_2 = NeuralProcess(4, 4, 12, 12, 6, 5, 'adam', 0.001)
        for epoch in range(epochs):
            # data, target_fraction=0.2
            # [x, y]; x: function input (n_points,len_x), y: function output (n_points, len_y)
            loss = model_stage_2.epoch(data)
            if epoch % 100 == 0:
                print('epoch: ',epoch,', loss = ',loss.numpy())
        self.stage_2_model = model_stage_2

    def fit_stage_3(self, data, epochs=1000):
        # x_dim, y_dim, layer_dim, r_dim, z_dim, num_samples, optimizer, learning_rate
        model_stage_3 = NeuralProcess(6, 4, 12, 12, 6, 5, 'adam', 0.001)
        for epoch in range(epochs):
            # data, target_fraction=0.2
            # [x, y]; x: function input (n_points,len_x), y: function output (n_points, len_y)
            loss = model_stage_3.epoch(data)
            if epoch % 100 == 0:
                print('epoch: ',epoch,', loss = ',loss.numpy())
        self.stage_3_model = model_stage_3