import numpy as np
import domain as do

esp = do.Domain_space(a = 0,b = 2, del_h= 0.01)

print(np.arange(esp.a,esp.b,esp.del_h))