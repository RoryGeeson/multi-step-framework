import numpy as np

class SobolGenerator():
    """"""

    def __init__(self, dimensions, num_points):

        self.dimensions = dimensions

        self.num_points = num_points

        self.sequence = np.zeros((self.dimensions,self.num_points))

    def

    def totient(self,n):
        y = n
        for i in range(2,n+1):
            if isPrime(i) and n % i == 0:
                y *= 1 - 1.0/i
        return int(y)

    def generatePolynomial(self, degree):
        """

        :param degree:
        :return:
        """

        return polynomial