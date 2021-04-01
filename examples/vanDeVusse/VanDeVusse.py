from math import exp
import numpy as np
from scipy.integrate import odeint, quad
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

    def filling_ODEs(self, species, t, P1, P2, P3, V0, Theta):
        """

        """
        _, _, CA, CB = species

        dCA_dTheta = -P1 * CA - P3 * CA ** 2 - CA / (V0 + Theta) + 1 / (V0 + Theta)

        dcB_dTheta = P1 * CA - P2 * CB - CB / (V0 + Theta)

        CA_out += CA * t
        CB_out += CB * t

        return CA_out, CB_out, dCA_dTheta, dcB_dTheta

    def draining_ODEs(self, species, t, P1, P2, P3):
        """

        """
        CA, CB = species

        dCA_dTheta = -P1 * CA - P3 * CA ** 2

        dcB_dTheta = P1 * CA - P2 * CB
        return dCA_dTheta, dcB_dTheta

    def noiseMaker(self, x, noise_scale=0.025):
        return np.random.normal(x,noise_scale*x)

    def stage_1(self, CA, P1, P2, P3, V0, Theta, t_out):
        t_out = np.linspace(0, t_out, 500)
        species = [CA, 0] # CA, CB
        CA, CB = odeint(self.filling_ODEs, species, t_out, (P1, P2, P3, V0, Theta))
        return CA, CB

    def stage_2(self, CA, CB, P1, P2, P3, t_out):
        t_out = np.linspace(0, t_out, 500)
        species = [CA, CB] # CA, CB
        CA, CB = odeint(self.draining_ODEs, species, t_out, (P1, P2, P3))
        # CA, CB = quad(lambda t: quad(self.draining_ODEs, 0, t, (P1, P2, P3))[0], 0, t_out)[0]
        return CA, CB

    def stage_3(self, CA, CB, P1, P2, P3, t_out):
        t_out = np.linspace(0, t_out, 500)
        species = [0, 0, CA, CB] # CA, CB
        CA_total, CB_total, _, _ = odeint(self.draining_ODEs, species, t_out, (P1, P2, P3))
        CA = CA_total / t_out
        CB = CB_total / t_out
        return CA, CB

    def noiselessExperimentalRun(self, CA, conditions):
        # [P1, P2, P3, V0, sigmaF]
        yieldList = []
        CA, CB = self.stage_1(CA, *conditions[0:5])
        yieldList += [CA, CB]
        CA, CB = self.stage_2(CA, CB, *conditions[0:3], 1 - 2 * conditions[4])
        yieldList += [CA, CB]
        CA, CB = self.stage_3(CA, CB, *conditions[0:3], conditions[4])
        yieldList += [CA, CB]
        return np.array(yieldList)

    def experimentalRun(self, CA, conditions):
        """
        Generates yields of an experimental run at given conditions, adding normally distributed noise to the results
        :param conditions: list of the experimental conditions [split, T1, P1, solvent, t_out1, T2, P2, t_out2, T3, P3, t_out3]
        :return: list of experimental yields [A1, A2, B, C, D, E, F]
        """
        noiselessData = np.array(self.noiselessExperimentalRun(CA, conditions)).clip(0)
        experimentData = self.noiseMaker(noiselessData)
        experimentData = [relu(data) for data in experimentData]
        return experimentData

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