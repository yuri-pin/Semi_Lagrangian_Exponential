import numpy as np
import domain as do
import functions as func
import auxiliar as auxi
import cond_initial as cond_ini
import methods as meth

def func_exact(ini,cond_ini,t,x = 0):
    if ini == 0:
      return np.exp(2*t)
    elif ini == 1:
      return np.exp(2*t)*t + np.exp(2*t)
    elif ini == 2:
      return np.cos(t) + np.exp(-1000*t)
    elif ini == 3:
      return (1+(1j/1001))*np.exp(-1000j*t) - 1j*np.exp(1j*t)/1001
    elif ini == 4 or ini == 5:
      return cond_ini.cond_ini_IE( x + np.sqrt(2)*t)


def sol_reference(init,spec,cond, M,dt):
    dom_esp = do.Domain_space(a = 0,b = 1,del_h = 1/M)
    
    
    #defining the time, if the \Delta t vary or not 
    dom_time = do.Domain_temp(ti= 0, tf = 0.15,del_t = dt)
    
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
    
    

    u_f_ref = meth.RungeKutta44(x=x_hat, dom_tem=dom_time, lin=lin_ref, non_lin=non_lin_ref, normal=normal_ref)



    # 5. Truncamento espectral para alinhar com os modos da malha de teste
    u_hat_final = u_f_ref[-1]
    N_line = (np.size(u_hat_final)-1)//2
    u_trunc, _ = auxiliar_test.trunc_f(u_hat_final, k_line, N_line)

    # 6. Transformada inversa para o espaço real
    return auxiliar_test.IFFT(u_trunc).real