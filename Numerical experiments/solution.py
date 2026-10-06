import numpy as np

def func_exact(ini,t):
    if ini == 0:
      return np.exp(2*t)
    elif ini == 1:
      return np.exp(2*t)*t + np.exp(2*t)
    elif ini == 2:
      return np.cos(t) + np.exp(-1000*t)
    elif ini == 3:
      return (1+(1j/1001))*np.exp(-1000j*t) - 1j*np.exp(1j*t)/1001