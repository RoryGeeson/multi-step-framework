import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
import tensorflow.keras.optimizers as optimizers
import matplotlib.pyplot as plt
# import datetime

class NeuralProcess(tf.keras.Model):
    """

    """
    def __init__(self, x_dim, y_dim, layer_dim, r_dim, z_dim, num_samples, optimizer, learning_rate):
        super(NeuralProcess,self).__init__()

        self.num_samples = num_samples

        self.x_dim = x_dim

        self.y_dim = y_dim

        self.layer_dim = layer_dim

        self.r_dim = r_dim

        self.z_dim = z_dim

        self.optimizer_dict = {
            'adadelta': optimizers.Adadelta(learning_rate),
            'adagrad': optimizers.Adagrad(learning_rate),
            'adam': optimizers.Adam(learning_rate),
            'adamax': optimizers.Adamax(learning_rate),
            'ftrl': optimizers.Ftrl(learning_rate),
            'nadam': optimizers.Nadam(learning_rate),
            'rmsprop': optimizers.RMSprop(learning_rate),
            'sgd': optimizers.SGD(learning_rate)
        }

        self.optimizer = self.optimizer_dict[optimizer]

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.x_dim+self.y_dim,)),
            tf.keras.layers.Dense(self.layer_dim, activation='elu'),
            tf.keras.layers.Dense(self.r_dim)
        ])

        self.mu_encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.r_dim)),
            tf.keras.layers.Dense(self.z_dim)
        ])

        self.sigma_encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.r_dim)),
            tf.keras.layers.Dense(self.z_dim)
        ])

        self.mu_decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.x_dim+self.z_dim,)),
            tf.keras.layers.Dense(self.layer_dim, activation='sigmoid'),
            tf.keras.layers.Dense(self.y_dim)
        ])

        self.sigma_decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.x_dim + self.z_dim,)),
            tf.keras.layers.Dense(self.layer_dim, activation='sigmoid'),
            tf.keras.layers.Dense(self.y_dim)
        ])

    def encode(self,x,y):
        """
        Encodes the x, y context pairs into the representation space r
        :param x: (n_points_c,len_x)
        :param y: (n_points_c,len_y)
        :return: r: (n_points_c,r_dim)
        """
        # [x,y]: [(n_points_c,len_x), (n_points_c,len_y)]
        # x_y: [n_points_c, len_x+len_y]
        # print(x)
        # print(y)
        x_y = tf.concat([x,y], axis=1)
        # r: [n_points_c,r_dim]
        r = self.encoder(x_y)
        return r

    def aggregate(self,r):
        """
        Aggregates the representation space vectors into a single distribution (z-space with z normally distributed)
        :param r: representation: (n_points_c,r_dim)
        :return: aggregated r value: (1,r_dim)
        """
        # Order-ambivalent aggregation of representations: (r_dim)
        r_aggregated = tf.reshape(tf.math.reduce_mean(r, 0),[1,-1])
        return r_aggregated

    def z_encoder(self, r_aggregated):
        # mu: [z_dim]
        mu = self.mu_encoder(r_aggregated)
        # sigma: [z_dim]
        sigma = 0.1 + 0.9 * tf.math.sigmoid(self.sigma_encoder(r_aggregated))
        z_distribution = tfp.distributions.Normal(mu, sigma)
        # z_params = [mu, sigma]
        return z_distribution

    def decode(self, x, z_distribution):
        """
        Decodes the z space representation along with the target x values into target y values
        :param x:
        :param z_params:
        :return:
        """
        # print('x: ',x)
        # x_mult: [num_x_train, num_samples, x_dim]
        # x_mult = tf.cast(tf.tile(tf.expand_dims(x,[1]),(1,self.num_samples,1)),dtype=tf.float32)
        # x_mult: [num_samples, num_x_train, x_dim]
        x_mult = tf.cast(tf.tile(tf.expand_dims(x,[0]),(self.num_samples,1,1)),dtype=tf.float32)
        # # z: [num_x_train, num_samples, z_dim]
        # # z_distribution.sample([num_samples]): (num_samples,1,z_dim)
        # z = tf.tile(tf.reshape(z_distribution.sample([self.num_samples]),[1,self.num_samples,self.z_dim]), (x_mult.shape[0], 1, 1))
        # z: [num_samples, num_x_train, z_dim]
        # z_distribution.sample([num_samples]): (num_samples,1,z_dim)
        z = tf.tile(z_distribution.sample([self.num_samples]), (1, x_mult.shape[1], 1))
        # x_z: [num_x_train, num_samples, x_dim+z_dim]
        x_z = tf.concat([x_mult, z], axis= 2)
        # print('num_samples: ',self.num_samples)
        # print('x_z: ',x_z)
        # x_z = tf.reshape(x_z,[-1,self.x_dim+self.z_dim])
        # print('x_z: ',x_z)
        # y_mu = tf.reshape(y_muu,[-1,self.num_samples])
        y_mu = self.mu_decoder(x_z)
        # y_sigma = tf.reshape(self.sigma_decoder(x_z),[-1,self.num_samples])
        y_sigma = self.sigma_decoder(x_z)
        # y: [num_x_train, y_dim]
        return y_mu, 0.1 + 0.9 * tf.math.softplus(y_sigma)

    def ELBO(self, z_distribution_c, z_distribution_all, train):
        """
        Calculates the Evidence Lower Bound loss with 'num_samples' samples
        :param z_params_c:
        :param z_params_all:
        :param dataset:
        :return:
        """
        y_mu, y_sigma = self.decode(train[0], z_distribution_all)
        # print('mu: ',y_mu)
        # print('sigma: ',y_sigma)
        # print('train: ',train[1])
        dist = tfp.distributions.Normal(y_mu, y_sigma)
        # print(dist.sample())
        log_probs = dist.log_prob(train[1])
        error = tf.reduce_mean(tf.reduce_sum(log_probs, axis=0))
        divergence = tfp.distributions.kl_divergence(z_distribution_all, z_distribution_c)
        divergence = tf.reduce_sum(divergence)
        loss = divergence - error
        return loss

    def loss(self, context, train, both):
        """
        Calculates the neural process loss function
        :param dataset: dictionary containing context, train, and both datasets for training in the given epoch
        :return: loss tensor (1)
        """
        # Shape of dataset[context]: [[(n_points_c,len_x)][(n_points_c,len_y)]]
        # r_c: [n_points, r_dim]
        r_c = self.encode(*context)
        # r_c_aggregated: [r_dim]
        r_c_aggregated = self.aggregate(r_c)
        # z_distribution: tfp distribution
        z_distribution_c = self.z_encoder(r_c_aggregated)
        r_all = self.encode(*both)
        r_all_aggregated = self.aggregate(r_all)
        self.z_distribution = self.z_encoder(r_all_aggregated)
        ELBO = self.ELBO(z_distribution_c, self.z_distribution, train)
        return ELBO

    def gradients(self, context, train, both):
        """
        Calculate gradient of objective w.r.t. parameters
        :param dataset: dictionary containing context, train, and both datasets for training in the given epoch
        :return: gradients
        """
        with tf.GradientTape() as tape:
            loss = self.loss(context, train, both)
        return tape.gradient(loss, self.trainable_variables), loss

    def optimize(self, gradients):
        optimizer = self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))
        return optimizer

    def epoch(self, data, target_fraction=0.2):
        """
        Perform an epoch.
        :param data: List containing [x, y]; x: function input (n_points,len_x), y: function output (n_points, len_y)
        :return: loss
        """
        x_c, x_t, y_c, y_t = train_test_split(data[0],data[1],test_size=target_fraction)
        context = [x_c,y_c]
        train = [x_t,y_t]
        both = [data[0],data[1]]
        gradients, loss = self.gradients(context, train, both)
        self.optimize(gradients)
        return loss

    def visualise(self, x, y, x_star, plot_var, stage):
        plt.figure(figsize=(8, 8))
        # r = self.encode(x,y)
        # r_aggregated = self.aggregate(r)
        # z_dist = self.z_encoder(r_aggregated)
        # mu, _ = self.decode(x_star, z_dist)
        mu, _ = self.decode(x_star, self.z_distribution)
        x_star = np.tile(x_star, (self.num_samples,1,1))
        for j in range(mu.shape[2]):
            for i in range(mu.shape[1]):
                plt.plot(x_star[:,i,plot_var], mu.numpy()[:, i, j], linewidth=1)
            plt.scatter(x[:,plot_var], y[:,j])
            # plt.title('Function Samples for New Input at Epoch {}'.format(epoch))
            plt.xlabel('x')
            plt.ylabel('y{}'.format(j))
            plt.savefig('C:/Users/geeso/OneDrive - University of Cambridge/Year_1/Pietro/multi-step/examples/Toy_problem/Plots/Stage_Plots/Stage_{}_y_{}.png'.format(stage,j),
                        bbox_inches='tight')
            plt.clf()
            plt.cla()
            plt.close()

    def evaluateModel(self, x):
        mu, _ = self.decode(tf.expand_dims(x,[0]), self.z_distribution)
        return tf.squeeze(tf.reduce_mean(mu, axis=0))

    def optimise_model(self, data, epochs=10):
        """"""
        # Define our metrics
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        # train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
        # test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

        # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        # test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        for epoch in range(epochs):
            split_numerator = np.random.randint(1, data[0].size-1)
            target_fraction = split_numerator / data[0].size
            loss = self.epoch(data, target_fraction)
            train_loss(loss)
            # train_accuracy(y_train, predictions)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                # tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

            # template = 'Epoch {}, Loss: {}'
            # print(template.format(epoch + 1,
            #                       train_loss.result()))
            # Reset metrics every epoch
            train_loss.reset_states()
            # test_loss.reset_states()
            # train_accuracy.reset_states()
            # test_accuracy.reset_states()

        # x_g = np.arange(-4,4, 0.1).reshape(-1, 1).astype(np.float32)
        # self.visualise(*data, x_g, epochs)