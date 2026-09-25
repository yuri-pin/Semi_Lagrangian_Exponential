import numpy as np
import matplotlib.pyplot as plt
import domain as do 
import cond_initial as cond_ini
import auxiliar as auxi
import functions as func
import methods as meth




init = 6
spec = 1
velo = 0

N = 10

Erro = np.zeros([3,N])
H_2  = np.zeros([3,N])

for j in range(3):
    cond = j #para determinar a função seno que não dá problema
    for i in range(N):
        M_spec = 2**(i)         # número de frequencias no espectral final
        espaco_ref = do.Domain_space(a=0,b = 1,del_h = 1/(4*M_spec + 1))
        x_ref = np.arange(espaco_ref.a,espaco_ref.b,espaco_ref.del_h)
        condicao_initial_ref = cond_ini.cond_initial(cond = cond,dom_esp = espaco_ref, auxi = auxi.aux())

        temp_ref = do.Domain_temp(ti=0,tf=0.15,del_t=0.1/(2**(i)))
        auxiliar_ref = auxi.aux(ini = init,spec = spec,dom_esp = espaco_ref,dom_tem = temp_ref)
        lin_ref = func.linear_term_SE(ini = init,spec = spec,dom_esp = espaco_ref,dom_tem = temp_ref, aux = auxiliar_ref)
        non_lin_ref = func.non_linear_term_SE(ini = init,spec = spec,dom_esp = espaco_ref,dom_tem = temp_ref, aux = auxiliar_ref)
        normal_ref = func.normal_term_SE(ini = init,spec = spec,dom_esp = espaco_ref,dom_tem = temp_ref, aux = auxiliar_ref)



        fun = meth.RungeKutta44

    

        u_0_line = condicao_initial_ref.cond_ini_SE(x=x_ref)

        u_hat_line, k_line = auxiliar_ref.FFT(u_0_line)
    

        u_f = fun(x = u_hat_line,dom_tem = temp_ref, lin = lin_ref,non_lin = non_lin_ref,
                normal = normal_ref)



        N_line = (np.size(u_f[-1,:])-1)//2
        k = np.zeros(len(u_f[-1,:]), dtype = int)
        u_trunc,k_trunc = auxiliar_ref.trunc_f(u_f[-1,:],k,N_line)




        u_sol = auxiliar_ref.IFFT(u_trunc)
        espaco_semilag = do.Domain_space(a=0,b = 1,del_h = 1/(2*M_spec+1))
        x_semilag = np.arange(espaco_semilag.a,espaco_semilag.b,espaco_semilag.del_h)

    plt.plot(x_semilag,u_sol.real)
    ##############################################################################################
    ################## Até aqui fizemos a solução exata que será usada para parametrizar##########
    ##############################################################################################





plt.savefig("/storage/yuriassis/SE_Integrador/Numerical experiments/teste1.png", dpi =200)

plt.show()
   
