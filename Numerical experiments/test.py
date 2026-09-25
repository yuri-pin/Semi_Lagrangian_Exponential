import numpy as np
import domain as do
import auxiliar as auxi 

esp = do.Domain_space(a = 0,b = 2, del_h= 0.1)
esp2 = do.Domain_space(a=0,b = 2, del_h=0.1/2)
auxi = auxi.aux()

x_i =np.arange(esp.a,esp.b,esp.del_h)
x = np.arange(esp2.a,esp2.b,esp2.del_h)
print(x_i)
print(x)

print(x_i - auxi.reestrut(n = 2,x_i = x_i,x=x))

