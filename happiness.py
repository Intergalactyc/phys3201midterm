# Elliott Walker

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

pathname = "./happiness.csv"

def gen_data(N=50, mu_I=65, sigma_I=40):
    # Generate income/health/happiness datapoints
        # Income: Cutoff Gaussian
        # Health: Random [1,7] + up to 3 based on income
        # Happiness: 2*income/mean+health/2+Gaussian noise, cut
    results = [["Income","Health","Happiness"]]
    for i in range(N):
        income = 0
        happiness = 0
        while income < 20:
            income = random.gauss(mu_I,sigma_I)
        health = random.randint(1,4)+2*(income > mu_I/2)+2*(income > mu_I)+2*(income > mu_I+sigma_I)
        while happiness < 1 or happiness > 10.5:
            happiness = 2 * income / mu_I + health / np.sqrt(2) + random.gauss(2,2)
        results.append([int(income),int(health),int(happiness)])
    return np.array(results)

def leastsquares(data, *, verbose=False):
    # General least-squares plane fit for dependent variable y with k indeps xk
    # Returns coefficients as tuple (a0,...,ak)
    data = data.astype(np.float64)
    A = np.concatenate((np.ones(shape=(len(data),1)), data[:,:-1]), axis=1)
    M = np.matmul(np.transpose(A),A)
    b = np.matmul(np.transpose(A),data[:,-1])
    Minv = np.linalg.inv(M)
    result = np.matmul(Minv, b)
    if verbose:
        print("Matrix M: ")
        print(M.astype(np.int32))
        print("Vector b: ")
        print(b.astype(np.int32))
        print("M inverse: ")
        print(Minv)
        print("Resulting vector of coefficients: ")
        print(result)
    return result

def hyperplane(params):
    # Takes parameters a0, a1, ..., ak and returns
    # the function y = a0 + a1x1 + ... + akxk
    def inner(*args):
        result = params[0]
        for i, arg in enumerate(args):
            result += params[i+1] * arg
        return result
    return inner

def correlation(data1,data2):
    n = len(data1)
    mu1 = np.mean(data1)
    mu2 = np.mean(data2)
    var1 = 0
    var2 = 0
    cov = 0
    for i in range(n):
        d1 = data1[i]-mu1
        d2 = data2[i]-mu2
        var1 += d1*d1
        var2 += d2*d2
        cov += d1*d2
    return cov/(np.sqrt(var1*var2))

def r_squared(data, *, verbose=False):
    # Compute coefficient of multiple correlation
    # Not the fastest way to do it because there are redundant calculations
    # but it works and is fast enough for this small problem
    k = data.shape[1]-1
    R = np.array([[correlation(data[:,i],data[:,j]) for i in range(k)] for j in range(k)])
    c = np.array([correlation(data[:,i],data[:,-1]) for i in range(k)])
    if verbose:
        print("Correl. matrix R: ")
        print(R)
        print("Correl. vector c: ")
        print(c)
    return np.matmul(np.matmul(np.transpose(c),np.linalg.inv(R)),c)

if __name__ == "__main__":
    # find or generate data
    from os.path import exists
    if exists(pathname):
        data = np.genfromtxt(pathname,delimiter=",")
    else:
        data = gen_data(30)
        df = pd.DataFrame(data)
        df.to_csv(pathname, header=False, index=False)
    data = data[1:,:].astype(np.float64)
    # call least squares and correlation functions
    result = leastsquares(data, verbose=True)
    rsq = r_squared(data, verbose=True)
    print(f"r^2 = {rsq}")
    # 3d scatter plot and plane
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(data[1:,0], data[1:,1], data[1:,2], color="blue", alpha=0.7)
    max_income = max(data[1:,0])
    x_income = np.linspace(start=0,stop=max_income,num=100)
    y_health = np.linspace(start=0,stop=11,num=100)
    X, Y = np.meshgrid(x_income, y_health)
    plane = hyperplane(result)
    ax.plot_surface(X, Y, plane(X,Y), color="red", alpha=0.35)
    ax.set_xlabel("Income ($1000/yr)")
    ax.set_ylabel("Health")
    ax.set_zlabel("Happiness")
    plt.show()
