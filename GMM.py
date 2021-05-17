import numpy as np
import math
import random
import pandas as pd

def EM(Data, num, iter):

    Thetalist = []

    def maxabsarr(arr):
        veclist = []

        for vec in range(np.shape(arr)[0]):

            veclist.append(np.linalg.norm(arr[vec]))

        return max(veclist)

    Theta = []

    for gaussian in range(num):

        mu = random.choice(Data)
        sigma = np.random.rand(len(Data[0]), len(Data[0])) * random.uniform(0, maxabsarr(Data))
        sigma = np.matmul(sigma, np.transpose(sigma))
        Theta.append([mu, sigma, 1/num])

    print('Initialise:')

    print(Theta)

    def Gaussian(x, Args):

        dim = len(x)

        devmu = x-Args[0]

        covInv = np.linalg.inv(Args[1])

        numerator = np.matmul(devmu, covInv)
        numerator = np.matmul(numerator, np.transpose(devmu))
        numerator = math.e**(-1/2 * numerator.item())

        denom = math.sqrt((2*math.pi)**dim * np.linalg.det(Args[1]))

        return numerator/denom

        #for distribution in range(len(sumprob)):
         #   sumprob[distribution] = sumprob[distribution] * Theta[distribution][2]

    def Expectation(obs, distribution, Theta):

        numerator = Theta[distribution][2]*Gaussian(obs, Theta[distribution])

        denominator = []

        for dist in Theta:

            denominator.append(dist[2]*Gaussian(obs, dist))

        return numerator/sum(denominator)


    for iteration in range(iter):

        print(f'Iteration {iteration+1} out of {iter} start:')

        ##For optimising pi

        for dist in range(num):

            sum_exp_pi = []

            for point in Data:

                #print(Gaussian(point, Theta[dist]))
                sum_exp_pi.append(Expectation(point, dist, Theta))

            pi = sum(sum_exp_pi)*1/len(Data)

            Theta[dist][2] = pi

        ##For optimising mu

        for dist in range(num):

            sum_exp_point_mu = []
            sum_exp_mu = []

            for point in Data:

                sum_exp_point_mu.append(point*Expectation(point, dist, Theta))
                sum_exp_mu.append(Expectation(point, dist, Theta))

            mu = sum(sum_exp_point_mu)/sum(sum_exp_mu)

            Theta[dist][0] = mu

        #For optimising cov

        for dist in range(num):

            sum_dev_sig = []
            sum_exp_sig = []

            for point in Data:


                dev = np.matmul(np.transpose([point-Theta[dist][0]]), [point-Theta[dist][0]])

                #print(dev)

                sum_dev_sig.append(Expectation(point, dist, Theta)*dev)

                sum_exp_sig.append(Expectation(point, dist, Theta))

            sig = sum(sum_dev_sig)/sum(sum_exp_sig)

            Theta[dist][1] = sig

        print(Theta)

        Thetalist.append(Theta)

    print('List of parameters:')
    print(Thetalist)

    return Thetalist

#Usage Example:
EM(np.random.rand(10,2)*10,2,10)