import numpy as np

def Gaussian(x,z,sigma=1,axis=None):
    return np.exp((-(np.linalg.norm(x-z, axis=axis)**2))/(2*sigma**2))

x = np.arange(1, 6)
y = [1,3,5,7,9]

print x.shape

ps_gaussian = np.arange(0.1, 1, 0.1)

for ps in ps_gaussian:
    for (i,j) in zip(x,y):
        print i,j, Gaussian(i,j, ps)
