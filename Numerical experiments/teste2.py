import numpy as np
import matplotlib.pyplot as plt
import domain as do 
import cond_initial as cond_ini
import auxiliar as auxi
import functions as func
import methods as meth

init = 1
spec = 0
velo = 0
cond = 1

dom_esp = do.Domain_space(a = 0,b = 1,del_h = 0.001)
dom_time = do.Domain_temp(ti= 0, tf = 1,del_t = 0.2)

t = np.arange(dom_time.ti, dom_time.tf+dom_time.del_t, dom_time.del_t)

auxiliar_test = auxi.aux(ini = init,spec = spec,dom_esp = dom_esp,dom_tem = dom_time)
condition_initial= cond_ini.cond_initial(cond = cond,dom_esp = dom_esp, auxi = auxiliar_test)
lin_ref = func.linear_term_IE(ini = init,spec = spec,dom_esp = dom_esp,dom_tem = dom_time, aux = auxiliar_test)
non_lin_ref = func.non_linear_term_IE(ini = init,spec = spec,dom_esp = dom_esp,dom_tem = dom_time, aux = auxiliar_test)
normal_ref = func.normal_term_IE(ini = init,spec = spec,dom_esp = dom_esp,dom_tem = dom_time, aux = auxiliar_test)

x_0 = condition_initial.cond_ini_IE()

x1 = meth.Euler_Explicito(x = x_0,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref )

x2 = meth.RungeKutta44(x = x_0,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref )

x3 = meth.Euler_Exp_Explicito(x = x_0,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref )

x4 = meth.RK2_Exp_Explicito1(x = x_0,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref )

x5 = meth.RK2_Exp_Explicito2(x = x_0,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref )


plt.plot(t,x1, label = "Euler_explicito")
plt.plot(t,x2, label = "RK44")
plt.plot(t,x3, label = "Euler_Exp")
plt.plot(t,x4, label = "ETDRK2_trap")
plt.plot(t,x5, label = "ETDRK2_mid")
plt.legend()
plt.savefig("/storage/yuriassis/SE_Integrador/Numerical experiments/teste1.png", dpi =200)
plt.show()