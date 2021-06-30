import numpy as np
import sobol_seq as sob
import tensorflow as tf
import cvxpy as cp
import cvxopt
# from cvxpylayers.tensorflow import CvxpyLayer

def determineNumVariables(model):
    """"""
    numVariables = model.trainable_variables
    return numVariables

class objective():

    def __init__(self, objectiveFunction, objectiveVariables):
        self.objectiveFunction = objectiveFunction
        self.objectiveVariables = objectiveVariables

class stage():

    def __init__(self, stageID, model, inputRanges, outputDimension=None, followingStages=None):
        self.stageID = stageID
        self.model = model
        self.inputRanges = inputRanges
        self.outputDimension = outputDimension
        # followingStages includes the stageID and the variables relevant to it
        self.followingStages = followingStages

def createHyperparameterLenDict(model):
    hyperparameterLenDict = {}
    for index, layer in enumerate(model.typeLayer):
        hyperparameterLenDict['layer{}'.format(index)] = []
        # print('dict: {}, {} '.format(layer.name,layer.weights))
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
        elif 'graph_conv' in layer.name:
            startInd = 0
            endInd = startInd + tf.reduce_prod(layerWeights[0])
            layer.weight = tf.reshape(hyperparameters[index, startInd:endInd], layerWeights[0])
            startInd = endInd
            endInd = startInd + tf.reduce_prod(layerWeights[1])
            layer.bias = tf.reshape(hyperparameters[index, startInd:endInd], layerWeights[1])

def scaleInputs(inputs, inputRanges):
    scaledInputs = tf.math.multiply(inputs, inputRanges[:,0]) + inputRanges[:,1]
    return scaledInputs

def flatten(flattenable):
    def flatten2(flattenable):
        for i in flattenable:
            if isinstance(i, list):
                for j in flatten2(i):
                    yield j
            else:
                yield i
    return list(flatten2(flattenable))

def calculateOffset(stageGraph, stageID):
    try:
        previousStages = stageGraph.stagePrevious[stageID]
    except:
        return 0
    offset = 0
    for stage in previousStages:
        offset += len(stage[1])
    return offset

def sobolSequenceConditions(condition_ranges, num_points, reverse=False):
    """"""
    dimensions = len(condition_ranges)
    sobol_sequence = sob.i4_sobol_generate(dimensions, num_points)
    lower_bounds = np.array([condition[0] for condition in condition_ranges])
    upper_bounds = np.array([condition[1] for condition in condition_ranges])
    ranges = upper_bounds - lower_bounds
    offset = np.tile(lower_bounds,(num_points,1))
    conditions = np.array(sobol_sequence * ranges + offset)
    if reverse:
        conditions = np.flip(conditions, axis=0)
    return conditions


########

# https://github.com/AvivNavon/pareto-hypernetworks/blob/master/phn/solvers.py
#
#             ||
#             ||
#             ||
#             \/
########

# class Solver:
#     def __init__(self, n_tasks):
#         super().__init__()
#         self.n_tasks = n_tasks
#
#     @abstractmethod
#     def get_weighted_loss(self, losses, ray, parameters=None, **kwargs):
#         pass
#
#     def __call__(self, losses, ray, parameters, **kwargs):
#         return self.get_weighted_loss(losses, ray, parameters, **kwargs)

# class EPOSolver():
#     """Wrapper over EPO
#     """
#
#     def __init__(self, n_tasks, n_params):
#         super().__init__(n_tasks)
#         self.solver = EPO(n_tasks=n_tasks, n_params=n_params)
#
#     def __call__(self, losses, ray, parameters=None, **kwargs):
#         assert parameters is not None
#         return self.solver.get_weighted_loss(losses, ray, parameters)

class EPOSolver():
    def __init__(self, n_tasks, eps):#, n_params):
        self.n_tasks = n_tasks
        self.eps = eps
        # self.n_params = n_params

    # def __call__(self, losses, ray, parameters):
    #     assert parameters is not None
    #     return self.get_weighted_loss(losses, ray, parameters)

    # @staticmethod
    # def _flattening(grad):
    #     # return torch.cat(tuple(g.reshape(-1, ) for i, g in enumerate(grad)), axis=0)
    #     return tf.concat(tuple(tf.reshape(g,[-1,]) for i, g in enumerate(grad)), axis=0)

    def __call__(self, losses, ray, parameters, tape):
        assert parameters is not None
        # print('ray: ',ray)
        ray = tf.squeeze(ray) # tf.transpose(ray)
        lp = EPOLP(m=self.n_tasks, r=ray.numpy(), eps=self.eps)

        # print(losses)
        # print(losses.ndim)

        # grads = []
        # grad = tf.scan(lambda loss: tape.gradient(loss,parameters), losses)

        # print(grad)
        # for i, loss in enumerate(losses):
        #     # g = torch.autograd.grad(loss, parameters, retain_graph=True)
        #     g = tape.gradient(loss, parameters)
        #     print('g: ',g)
        #     # flat_grad = self._flattening(g)
        #     # print('flat_grad: ',flat_grad)
        #     # grads.append(flat_grad)
        #     grads.append(g)
        # print('losses: ',losses)
        # print('parameters: ',parameters)
        Jac = tape.jacobian(losses, parameters, experimental_use_pfor=False)
        shapes = []
        G = np.empty([self.n_tasks,0])
        for grads in Jac:
            shapes.append(grads.shape)
            G = np.append(G,tf.reshape(grads, [self.n_tasks, -1]).numpy(), axis=1)
        # Jac = [tf.reshape(grads,[2,-1]).numpy() for grads in Jac]
        # G = tf.reshape(Jac[0],[2,-1])
        # for grads in Jac[1:]:
        #     G = tf.concat([G,tf.reshape(grads,[2,-1])])
        # for row in G:
        #     print(tf.shape(row))
        # G = tape.gradient(losses, parameters)

        # G = torch.stack(grads)
        # G = tf.stack(grads)
        # print(Jac)
        GG_T = G @ G.T
        # GG_T = GG_T.numpy()
        # GG_T = GG_T.detach().cpu().numpy()

        # numpy_losses = losses.detach().cpu().numpy()
        numpy_losses = tf.squeeze(tf.transpose(losses)).numpy()

        try:
            alpha = lp.get_alpha(numpy_losses, G=GG_T, C=True)
        except Exception as excep:
            print(excep)
            alpha = None
        # print('alpha1: ',alpha)
        if alpha is None:  # A patch for the issue in cvxpy
            alpha = (ray / tf.reduce_sum(ray)).numpy()

        alpha *= self.n_tasks
        # alpha = torch.from_numpy(alpha).to(losses.device)
        alpha = tf.reshape(tf.convert_to_tensor(alpha, dtype=tf.float32),[self.n_tasks,1])
        # print('losses: ',losses)
        # print('alpha: ',alpha)
# Return weighted gradients instead of a weighted loss, will probably speed up
        weighted_loss = tf.reduce_sum(losses * alpha)
        # print('weighted_loss: ',weighted_loss)
        # gradients = tape.gradient(weighted_loss, parameters)
        return weighted_loss
        # print(len(parameters))
        # for item in parameters:
        #     print(item.shape)
        # print(alpha)
        # print(len(Jac))
        # for item in Jac:
        #     print(item.shape)
        # print(G)
        # # print('Jac0: ',Jac[0])
        # # print('Jac1: ',Jac[1])
        # G = G * alpha
        # print(G)
        # weighted_gradients = Jac * alpha
        # return weighted_gradients

class EPOLP():

    def __init__(self, m, r, eps=1e-4):
        # cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.m = m
        self.r = r
        self.eps = eps
        self.last_move = None
        self.a = cp.Parameter(m)  # Adjustments
        self.C = cp.Parameter((m, m))  # C: Gradient inner products, G^T G
        self.Ca = cp.Parameter(m)  # d_bal^TG
        self.rhs = cp.Parameter(m)  # RHS of constraints for balancing

        self.alpha = cp.Variable(m)  # Variable to optimize

        obj_bal = cp.Maximize(self.alpha @ self.Ca)  # objective for balance
        constraints_bal = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Simplex
                           self.C @ self.alpha >= self.rhs]
        self.prob_bal = cp.Problem(obj_bal, constraints_bal)  # LP balance
        # self.prob_bal = CvxpyLayer(obj_bal, constraints_bal)  # LP balance

        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C))  # obj for descent
        constraints_res = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Restrict
                           self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
                           self.C @ self.alpha >= 0]
        constraints_rel = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Relaxed
                           self.C @ self.alpha >= 0]
        self.prob_dom = cp.Problem(obj_dom, constraints_res)  # LP dominance
        self.prob_rel = cp.Problem(obj_dom, constraints_rel)  # LP dominance
        # self.prob_dom = CvxpyLayer(obj_dom, constraints_res)  # LP dominance
        # self.prob_rel = CvxpyLayer(obj_dom, constraints_rel)  # LP dominance

        self.gamma = 0  # Stores the latest Optimum value of the LP problem
        self.mu_rl = 0  # Stores the latest non-uniformity

    def get_alpha(self, l, G, r=None, C=False, relax=False):
        r = self.r if r is None else r
        assert len(l) == len(G) == len(r) == self.m, "length != m"
        rl, self.mu_rl, self.a.value = adjustments(l, r)
        self.C.value = G if C else G @ G.T
        self.Ca.value = self.C.value @ self.a.value

        if self.mu_rl > self.eps:
            J = self.Ca.value > 0
            if len(np.where(J)[0]) > 0:
                J_star_idx = np.where(rl == np.max(rl))[0]
                self.rhs.value = self.Ca.value.copy()
                self.rhs.value[J] = -np.inf  # Not efficient; but works.
                self.rhs.value[J_star_idx] = 0
            else:
                self.rhs.value = np.zeros_like(self.Ca.value)
            self.gamma = self.prob_bal.solve(solver=cp.GLPK, verbose=False)
            # self.gamma = CvxpyLayer(self.prob_bal, parameters=, variables=)
            # self.gamma = self.prob_bal.solve(verbose=False)
            self.last_move = "bal"
        else:
            if relax:
                self.gamma = self.prob_rel.solve(solver=cp.GLPK, verbose=False)
                # self.gamma = CvxpyLayer(self.prob_rel, parameters=, variables=)
            else:
                self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
                # self.gamma = CvxpyLayer(self.prob_dom, parameters=, variables=)
            # self.gamma = self.prob_dom.solve(verbose=False)
            self.last_move = "dom"

        return self.alpha.value

def mu(rl, normed=False):
    if len(np.where(rl < 0)[0]):
        raise ValueError(f"rl<0 \n rl={rl}")
        return None
    m = len(rl)
    l_hat = rl if normed else rl / rl.sum()
    eps = np.finfo(rl.dtype).eps
    l_hat = l_hat[l_hat > eps]
    return np.sum(l_hat * np.log(l_hat * m))

def adjustments(l, r=1):
    m = len(l)
    rl = r * l
    l_hat = rl / rl.sum()
    mu_rl = mu(l_hat, normed=True)
    a = r * (np.log(l_hat * m) - mu_rl)
    return rl, mu_rl, a