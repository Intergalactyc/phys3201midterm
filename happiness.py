import matplotlib.pyplot as plt
import pandas as pd
import random

def gen_data(N=50, mu_I=65, sigma_I=40):
    results = [["Income","Health","Happiness"]]
    for i in range(N):
        income = 0
        happiness = 0
        while income < 20:
            income = random.gauss(mu_I,sigma_I)
        health = random.randint(1,7)+(income > mu_I/2)+(income > mu_I)+(income > mu_I+sigma_I)
        while happiness < 1 or happiness > 10.5:
            happiness = 2 * income / mu_I + health / 2 + random.gauss(2,3)
        results.append([int(income),int(health),int(happiness)])
    return np.array(results)

def leastsquares(data):


data = gen_data(30)
df = pd.DataFrame(data)
df.to_csv("./happiness.csv", header=False, index=False)
