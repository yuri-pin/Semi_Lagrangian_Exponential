import numpy as np
import matplotlib.pyplot as plt
import domain as do 
import cond_initial as cond_ini
import auxiliar as auxi
import functions as func
import methods as meth
import solution as sol


titles_list = ("Error Behavior - Burger Equation - $\Delta t = 0.00001$",
               "Error Behavior - Burger Equation - $\Delta x = 1/65$",
               "Error Behavior - Burger Equation - $\Delta x = 1/65$",
               "Error Behavior - Burger Equation linear stiff term")

file_name = ("Error Behavior_Burger Equation_0.00001$",
               "Error Behavior_Burger Equation_1_65",
               "Error Behavior_Burger Equation_1_65_(1)",
               "Error Behavior_Burger Equation linear stiff term")

init_list = (6,6,7,7)
cond_list = (4,4,4,4)

del_h_list = (0,
              32,
              32,
              0)


del_t_list = (0.00001,
                0.15/(2**5),
                0.15/(2**5),
                0.15)

for j in range(1,len(init_list)):
    plt.figure() 

    init = init_list[j]
    spec = 1
    velo = 0
    cond = cond_list[j]

    M = 8

    valx1 = np.zeros(M,dtype=complex)
    valx2 = np.zeros(M,dtype=complex)
    valx3 = np.zeros(M,dtype=complex)
    valx4 = np.zeros(M,dtype=complex)
    valx5 = np.zeros(M,dtype=complex)
    valex = np.zeros(M,dtype=complex)
    H = np.zeros(M)

    for i in range(M):
        #construction of reference solution
        if j ==0 or j==3:
            M_spec = 2**(i+1)
        else:
            M_spec = del_h_list[j]

        if j == 0:
            dt = del_t_list[j]
        else:
            dt = del_t_list[j]/(2**(i+2)) 

        solution = sol.sol_reference(init = init, spec = spec, cond = cond,M = 4*M_spec+1, dt = dt)

        ##############################################
        ##############################################


        #defining the space, if the \Delta x vary or not 
        dom_esp = do.Domain_space(a = 0,b = 1,del_h = 1/(4*M_spec+1))


        #defining the time, if the \Delta t vary or not 
        if j!=0:
            dom_time = do.Domain_temp(ti= 0, tf = 0.15,del_t = del_t_list[j]/2**i)
        else:
            dom_time = do.Domain_temp(ti= 0, tf = 0.15,del_t = del_t_list[j])

        
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

        
        H[i] = dom_time.del_t

        iter = (j != 1) #special plot for the first case




        x1 = meth.Euler_Explicito(x = x_hat,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref )
        x2 = meth.RungeKutta22(x = x_hat,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref )
        x3 = meth.Euler_Exp_Explicito(x = x_hat,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref, iter  = iter)
        x4 = meth.RK2_Exp_Explicito1(x = x_hat,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref )
        x5 = meth.RK2_Exp_Explicito2(x = x_hat,dom_tem= dom_time, lin=lin_ref, non_lin=non_lin_ref, normal= normal_ref )

        
        N_line = (len(x1[-1]))/2
        y1,_ = auxiliar_test.trunc_f(x1[-1],k_line,N_line)
        y2,_ = auxiliar_test.trunc_f(x2[-1],k_line,N_line)       
        y3,_ = auxiliar_test.trunc_f(x3[-1],k_line,N_line)
        y4,_ = auxiliar_test.trunc_f(x4[-1],k_line,N_line)
        y5,_ = auxiliar_test.trunc_f(x5[-1],k_line,N_line)

        

        valx1[i] = auxiliar_test.erro2(solution,auxiliar_test.IFFT(y1).real, dom_esp.del_h)
        valx2[i] = auxiliar_test.erro2(solution,auxiliar_test.IFFT(y2).real, dom_esp.del_h)
        valx3[i] = auxiliar_test.erro2(solution,auxiliar_test.IFFT(y3).real, dom_esp.del_h)
        valx4[i] = auxiliar_test.erro2(solution,auxiliar_test.IFFT(y4).real, dom_esp.del_h)
        valx5[i] = auxiliar_test.erro2(solution,auxiliar_test.IFFT(y5).real, dom_esp.del_h)
        

    ### graph plots
    plt.plot(H,valx1, label = "Euler_explicito")
    plt.plot(H,valx2, label = "RK22")
    plt.plot(H,valx3, label = "Euler_Exp")
    plt.plot(H,valx4, label = "ETDRK2_trap")
    plt.plot(H,valx5, label = "ETDRK2_mid")

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