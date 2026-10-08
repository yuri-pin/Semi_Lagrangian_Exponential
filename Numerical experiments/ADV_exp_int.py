import numpy as np
import matplotlib.pyplot as plt
import domain as do 
import cond_initial as cond_ini
import auxiliar as auxi
import functions as func
import methods as meth
import solution as sol


titles_list = ("Error Behavior - Advection Equation",
               "Error Behavior - Advection Equation - $\Delta x = 1/21$",
               "Error Behavior - Advection Equation - $\Delta x = 1/41$",)

file_name = ("Error Behavior_Advection Equation",
               "Error Behavior_Advection Equation1_21",
               "Error Behavior_Advection Equation1_41")

init_list = (4,5,5)
cond_list = (4,5,5)

del_h_list = (1/((10*2)+1),
              1/((10*2)+1),
              1/((10*2*2)+1))


del_t_list = (0.01,
                0.01,
                0.01)

for j in range(len(init_list)):
    plt.figure() 

    init = init_list[j]
    spec = 1
    velo = 0
    cond = cond_list[j]

    M = 10
    valx1 = np.zeros(M,dtype=complex)
    valx2 = np.zeros(M,dtype=complex)
    valx3 = np.zeros(M,dtype=complex)
    valx4 = np.zeros(M,dtype=complex)
    valx5 = np.zeros(M,dtype=complex)
    valex = np.zeros(M,dtype=complex)
    H = np.zeros(M)

    for i in range(M):
        #defining the space, if the \Delta x vary or not 
        dom_esp = do.Domain_space(a = 0,b = 1,del_h = del_h_list[j])


        #defining the time, if the \Delta t vary or not 
        dom_time = do.Domain_temp(ti= 0, tf = 1,del_t = del_t_list[j]/2**i)


        
        x_ref = np.arange(dom_esp.a, dom_esp.b, dom_esp.del_h)
        t = np.arange(dom_time.ti, dom_time.tf+dom_time.del_t, dom_time.del_t)


        auxiliar_test = auxi.aux(ini = init,spec = spec,dom_esp = dom_esp,dom_tem = dom_time)
        condition_initial= cond_ini.cond_initial(cond = cond,dom_esp = dom_esp, auxi = auxiliar_test)
        lin_ref = func.linear_term_IE(ini = init,spec = spec,dom_esp = dom_esp,dom_tem = dom_time, aux = auxiliar_test)
        non_lin_ref = func.non_linear_term_IE(ini = init,spec = spec,dom_esp = dom_esp,dom_tem = dom_time, aux = auxiliar_test)
        normal_ref = func.normal_term_IE(ini = init,spec = spec,dom_esp = dom_esp,dom_tem = dom_time, aux = auxiliar_test)


        #defining the initial condition
        x_0 = condition_initial.cond_ini_IE(x = x_ref)

        #apply the Fourier transformation
        x_hat, k_line = auxiliar_test.FFT(x_0)



        x_exact = sol.func_exact(ini = init,cond_ini= condition_initial,t = dom_time.tf,x = x_ref)

        

        
        
        H[i] = dom_time.del_t


        x1 = meth.Euler_Explicito(x = x_hat,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref )
        x2 = meth.RungeKutta44(x = x_hat,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref )
        x3 = meth.Euler_Exp_Explicito(x = x_hat,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref, iter  = iter)
        x4 = meth.RK2_Exp_Explicito1(x = x_hat,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref )
        x5 = meth.RK2_Exp_Explicito2(x = x_hat,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref )


        
        valx1[i] = auxiliar_test.erro2(x_exact,auxiliar_test.IFFT(x1[-1]).real, dom_esp.del_h)
        valx2[i] = auxiliar_test.erro2(x_exact,auxiliar_test.IFFT(x2[-1]).real, dom_esp.del_h)
        valx3[i] = auxiliar_test.erro2(x_exact,auxiliar_test.IFFT(x3[-1]).real, dom_esp.del_h)
        valx4[i] = auxiliar_test.erro2(x_exact,auxiliar_test.IFFT(x4[-1]).real, dom_esp.del_h)
        valx5[i] = auxiliar_test.erro2(x_exact,auxiliar_test.IFFT(x5[-1]).real, dom_esp.del_h)


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

