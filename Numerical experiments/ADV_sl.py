import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

import domain as do
import auxiliar as auxi
import cond_initial as cond
import interpol as inte
import profile_vel as prof_vel
import find_dep as fd
import methods as met


title_list = ("Error x $\Delta x$ - Linear Interpolation",
              "Error x $\Delta x$ - Cubic Interpolation",
              "Error x $\Delta x$ - Linear Interpolation - Proportional",
              "Error x $\Delta x$ - Cubic Interpolation - Proportional",
              "Error x $\Delta x$ - Linear Interpolation",
              "Error x $\Delta x$ - Cubic Interpolation",
              "Error x $\Delta x$ - Linear Interpolation - Proportional",
              "Error x $\Delta x$ - Cubic Interpolation - Proportional")

file_name = ("Error_Behavior_SL_Advetion_Linear",
             "Error_Behavior_SL_Advetion_Cubic",
             "Error_Behavior_SL_Advetion_Linear_proportional",
             "Error_Behavior_SL_Advetion_Cubic_propotional",
             "Error_Behavior_SL_Advetion_Linear",
             "Error_Behavior_SL_Advetion_Cubic",
             "Error_Behavior_SL_Advetion_Linear_proportional",
             "Error_Behavior_SL_Advetion_Cubic_propotional")

cond_list = (4,6,7,8)
vel_list = (0,0,0,0,1,1,1,1,5,6)

interp_list = ("P1","P3","P1","P3",
               "P1","P3","P1","P3")


name_functions = ("sinusoidal",
                  "gaussian",
                  "triangular",
                  "square")

pv_list = (2**(1/2),2**(1/2),2**(1/2),2**(1/2),
           1,1,1,1)

del_h_list = (1/((10)),1/((10)),1/((10)),1/((10)),1/((100)),1/((100)),1/((10)),1/((10)))


del_t_list = (0.01,0.01,1,1,
              0.01,0.01,1,1,)

m = 10
for j in range(len(title_list)):
    print(j)
    plt.figure()
    Error = np.zeros((4,m))
    

    for i in range(len(cond_list)):
        H = np.zeros(m)
        for k in range(m):
            espace = do.Domain_space(a = 0,b = 1, del_h=del_h_list[j]*2**(-k-2))
            if j in (0,1,4,5):
                time = do.Domain_temp(ti = 0, tf= 1, del_t = del_t_list[j])
            else:
                time = do.Domain_temp(ti = 0, tf = 1,del_t=del_t_list[j]*2**(-k-2))

            auxiliar = auxi.aux()

            cond_init = cond.cond_initial(cond = cond_list[i],dom_esp= espace)
            find_depar = fd.find_departure(dom_esp=espace,dom_temp= time)

            inter = inte.interpol(dom_esp=espace ,dom_temp = time )

            inter_di = {'P1': inter.P1, 'P3': inter.P3}

            velo = prof_vel.perf_vel(dom_esp=  espace, dom_temp= time, vel=vel_list[j])
            

            x_espace = np.arange(espace.a,espace.b,espace.del_h)

            f_exact = cond_init.cond_ini_general(x = (x_espace -pv_list[j]*time.tf)%(espace.b - espace.a))

            f_f = met.adv(dom_esp= espace,dom_temp= time, inter_met=inter_di[interp_list[j]],
                          find_d=find_depar.SETTLS, inter_ad= inter.P3, cond= cond_init.cond_ini_general(x = x_espace),
                          vel_0= velo.profile,x_i= x_espace)

            Error[i,k] = auxiliar.erro2(x = f_exact,y = f_f,dx = espace.del_h)
            H[k] = espace.del_h

    if j in (0,1,2,3):
        plt.plot(H,Error[0,:],"o-",label = name_functions[0])
        plt.plot(H,Error[1,:],"o-",label = name_functions[1])
        plt.plot(H,Error[2,:],"o-",label = name_functions[2])
        plt.plot(H,Error[3,:],"o-",label = name_functions[3])

        plt.loglog()

        plt.title(title_list[j])
        plt.legend()


        name = "/storage/yuriassis/SE_Integrador/Numerical experiments/" + file_name[j] + ".png"
        plt.savefig(name, dpi =200)
        plt.show()


        plt.close() 
    elif j in (4,5,6,7):
        name1 = "/storage/yuriassis/SE_Integrador/Numerical experiments/" + file_name[j] + ".txt"

        complete_data = np.column_stack((H,Error.T))

        header = ["$\Delta x$", name_functions[0],name_functions[1],name_functions[2],name_functions[3]]

        table = tabulate(complete_data, headers= header, tablefmt="grid", floatfmt=".16e")

        with open(name1, "w", encoding="utf-8") as arquivo: arquivo.write(table)