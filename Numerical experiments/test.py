import numpy as np
import matplotlib.pyplot as plt
import domain as do
import cond_initial as cond
import interpol as inte
import profile_vel as prof_vel
import find_dep as fd


def adv(dom_esp, dom_temp, inter_met, find_d, inter_ad, cond, vel_0, x_i):
    '''Advecção com velocidade variável no espaço e no tempo.'''
    f_old = cond

    u_old = vel_0(x_i, t = dom_temp.ti - dom_temp.del_t)[0]
    u_new = vel_0(x_i, t = dom_temp.ti)[0]

    for i in range(int((dom_temp.tf - dom_temp.ti) / dom_temp.del_t)):
        u_p_alp = find_d(dom_esp, dom_temp, inter_ad, x_i, u_new, u_old, M=5)

        f_new = inter_met(u_p_alp[2], u_p_alp[1], f_old)

        f_old = np.copy(f_new)
        u_old = np.copy(u_new)
        u_new = vel_0(x_i, t = dom_temp.ti + dom_temp.del_t * (i + 1))[0]

    return f_new


espace = do.Domain_space(a = 0,b = 1, del_h=0.0001)
time = do.Domain_temp(ti = 0, tf= 1, del_t = 0.001)

cond_init = cond.cond_initial(cond = 4,dom_esp= espace)
find_depar = fd.find_departure(dom_esp=espace,dom_temp= time)

inter = inte.interpol(dom_esp=espace ,dom_temp = time )
velo = prof_vel.perf_vel(dom_esp=  espace, dom_temp= time, vel=0)

x_espace = np.arange(espace.a,espace.b,espace.del_h)
 
f_f = adv(dom_esp= espace,dom_temp= time, inter_met=inter.P3,
          find_d=find_depar.SETTLS, inter_ad= inter.P3, cond= cond_init.cond_ini_general(x = x_espace),
          vel_0= velo.profile,x_i= x_espace)


plt.plot(x_espace,f_f)
plt.savefig("/storage/yuriassis/SE_Integrador/Numerical experiments/teste1.png", dpi =200)

plt.show()
