import numpy as np
import matplotlib.pyplot as plt
import domain as do 
import cond_initial as cond_ini
import auxiliar as auxi
import functions as func
import methods as meth
import solution as sol

titles_list = ("Error Behavior - Linear",
               "Error Behavior - Linear",
               "Error Behavior - Nonlinear",
               "Error Behavior - Nonlinear and stiff",
               "Error Behavior - Nonlinear and stiff")

file_name = ("Error Behavior - Linear",
            "Error Behavior - Linear1",
            "Error Behavior - Nonlinear",
            "Error Behavior - Nonlinear and stiff",
            "Error Behavior - Nonlinear and stiff1")

init_list = (0,0,1,2,3)

for j in range(len(init_list)):
    plt.figure() 

    init = init_list[j]
    spec = 0
    velo = 0
    cond = init

    M = 5
    valx1 = np.zeros(M,dtype=complex)
    valx2 = np.zeros(M,dtype=complex)
    valx3 = np.zeros(M,dtype=complex)
    valx4 = np.zeros(M,dtype=complex)
    valx5 = np.zeros(M,dtype=complex)
    valex = np.zeros(M,dtype=complex)
    H = np.zeros(M)

    #Generating the error of exact solution 
    for i in range(M):
        dom_esp = do.Domain_space(a = 0,b = 1,del_h = 0.001)
        dom_time = do.Domain_temp(ti= 0, tf = 1,del_t = 0.1/2**i)

        t = np.arange(dom_time.ti, dom_time.tf+dom_time.del_t, dom_time.del_t)

        auxiliar_test = auxi.aux(ini = init,spec = spec,dom_esp = dom_esp,dom_tem = dom_time)
        condition_initial= cond_ini.cond_initial(cond = cond,dom_esp = dom_esp, auxi = auxiliar_test)
        lin_ref = func.linear_term_IE(ini = init,spec = spec,dom_esp = dom_esp,dom_tem = dom_time, aux = auxiliar_test)
        non_lin_ref = func.non_linear_term_IE(ini = init,spec = spec,dom_esp = dom_esp,dom_tem = dom_time, aux = auxiliar_test)
        normal_ref = func.normal_term_IE(ini = init,spec = spec,dom_esp = dom_esp,dom_tem = dom_time, aux = auxiliar_test)

        x_0 = condition_initial.cond_ini_IE()


        x_exact = sol.func_exact(ini = init,cond_ini= condition_initial,t = dom_time.tf)
        
        H[i] = dom_time.del_t

        iter = (j != 1) #special plot for the first case

        x1 = meth.Euler_Explicito(x = x_0,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref )
        valx1[i] = abs(np.real(x1[-1]-x_exact)).item()
        x2 = meth.RungeKutta44(x = x_0,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref )
        valx2[i] = abs(np.real(x2[-1]-x_exact)).item()
        x3 = meth.Euler_Exp_Explicito(x = x_0,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref, iter  = iter)
        valx3[i] = abs(np.real(x3[-1]-x_exact)).item()
        x4 = meth.RK2_Exp_Explicito1(x = x_0,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref )
        valx4[i] = abs(np.real(x4[-1]-x_exact)).item()
        x5 = meth.RK2_Exp_Explicito2(x = x_0,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref )
        valx5[i] = abs(np.real(x5[-1]-x_exact)).item()


    ### graph plots
    plt.plot(H,valx1,"o-", label = "Euler_explicito")
    plt.plot(H,valx2,"o-", label = "RK44")
    plt.plot(H,valx3,"o-", label = "Euler_Exp")
    plt.plot(H,valx4,"o-", label = "ETDRK2_trap")
    plt.plot(H,valx5,"o-", label = "ETDRK2_mid")

    #scale
    plt.loglog()

    #legends and title
    plt.title(titles_list[j])
    plt.legend()


    #name
    name = "/storage/yuriassis/SE_Integrador/Numerical experiments/" + file_name[j] + ".png"
    plt.savefig(name, dpi =200)
    plt.show()


    plt.close() 



