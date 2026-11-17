import numpy as np

class find_departure:
  '''Essa classe determina qual será o perfil de velocidade que será usado no
  problema, sendo esse perfil constante ou não'''
  def __init__(self,dom_esp,dom_temp):
    self.dom_temp = dom_temp
    self.dom_esp = dom_esp


  def achar_departure_1(self,dom_esp,dom_temp,inter,x,u_1,u_0, M = 5):
    """dom é o dominio de nosso problema.
      f é o perfil de velocidade de nosso problema.
      x são os pontos que serão usados em nosso problema.
      precisaremos de dois perfis de velocidade o primeiro será u_0 e o delta t a frente será u_1"""
    #os campos de velocidades no tempo -deltat e 0 serão iguais só por conveniência.
    u_para = np.copy(u_1)
    u_old = np.copy(u_0)
    u_new = np.copy(u_1)
    #agora temos que achar o valor da extrapolação de u no tempo +1/2
    u_extra = (3*u_new - u_old)*0.5 #esse termo é para podermos fazer a interpolação
    u_extra1 = np.copy(u_extra) # esses e o termo seguinte são para iterar e fazer o cálculo do erro


    for i in range(M):#aqui atualizo o ponto fixo
      # Use the real part of u_extra1 for spatial shifts
      m = (u_extra1.real*self.dom_temp.del_t/(2*self.dom_esp.del_h))//1
      gama = (u_extra1.real*self.dom_temp.del_t/(2*self.dom_esp.del_h))%1
      u_extra1 = inter(gama,m,u_extra)

    # Use the real part of u_extra1 for spatial shifts
    p = (u_extra1.real*dom_temp.del_t/(dom_esp.del_h))//1
    alp = (u_extra1.real*dom_temp.del_t/(dom_esp.del_h))%1

    return np.array([u_extra,p,alp])

  def SETTLS(self,dom_esp,dom_temp,inter,x,u_1,u_0, M = 5):
    """dom é o dominio de nosso problema.
      f é o perfil de velocidade de nosso problema.
      x são os pontos que serão usados em nosso problema.
      precisaremos de dois perfis de velocidade o primeiro será u_0 e o delta t a frente será u_1"""
    #o paupite de depature point será o mesmo do arrival
    m = np.zeros_like(x)
    gama = np.zeros_like(x)
    #os campos de velocidades no tempo -deltat e 0 serão iguais só por conveniência.
    u_old = np.copy(u_0) #tempo t-deltat
    u_new = np.copy(u_1) #tempo t


    for i in range(M):#aqui atualizo o ponto fixo
      vd_old = inter(gama,m,u_old)
      vd_new = inter(gama,m,u_new)
      va_new = np.copy(u_new)

      settls = (2*vd_new - vd_old + va_new)*0.5

      # Use the real part of settls for spatial shifts
      m = (settls.real*self.dom_temp.del_t/(self.dom_esp.del_h))//1
      gama = (settls.real*self.dom_temp.del_t/(self.dom_esp.del_h))%1


    # Use the real part of settls for spatial shifts
    p = (settls.real*self.dom_temp.del_t/(self.dom_esp.del_h))//1
    alp = (settls.real*self.dom_temp.del_t/(self.dom_esp.del_h))%1
    return np.array([settls,p,alp])