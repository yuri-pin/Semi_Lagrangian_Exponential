import numpy as np




def Euler_Explicito(x,dom_tem,lin, non_lin,normal,impli = False,tol = 10**(-5)):
  '''Essa função tem como entradas os parametros:
  x --- condição inicial
  a --- ponto inicial
  b --- ponto final
  h --- passo no tempo
  f --- função a ser aplicada'''
  ############################################################
  a = dom_tem.ti
  b = dom_tem.tf
  h = dom_tem.del_t
  ############################################################

  x_total = np.zeros([int((b-a)/h)+1,len(x)],dtype=complex)
  x_total[0] = np.copy(x)
  for i in range(int((b-a)/h)):
    x_total[i+1] = x_total[i] + h*normal.normal(x_total[i],i*h)
  return x_total

def RungeKutta22(x,dom_tem,lin, non_lin,normal,impli = False,tol = 10**(-5)):
  '''Essa função tem como entradas os parametros:
  x --- condição inicial
  a --- ponto inicial
  b --- ponto final
  h --- passo no tempo
  f --- função a ser aplicada'''
  ############################################################
  a = dom_tem.ti
  b = dom_tem.tf
  h = dom_tem.del_t
  ############################################################

  x_total = np.zeros([int((b-a)/h)+1,len(x)],dtype=complex)
  x_total[0] = np.copy(x)
  for i in range(int((b-a)/h)):
    k1 = h*normal.normal(x_total[i],i*h)
    k2 = h*normal.normal(x_total[i]+k1/2,(i+0.5)*h)
    x_total[i+1] = x_total[i] + k2
  return x_total

def RungeKutta44(x,dom_tem,lin, non_lin,normal,impli = False,tol = 10**(-5)):
  '''Essa função tem como entradas os parametros:
  x --- condição inicial
  a --- ponto inicial
  b --- ponto final
  h --- passo no tempo
  f --- função a ser aplicada'''
  ############################################################
  a = dom_tem.ti
  b = dom_tem.tf
  h = dom_tem.del_t
  ############################################################

  x_total = np.zeros([int((b-a)/h)+1,len(x)],dtype=complex)
  x_total[0] = np.copy(x)
  for i in range(int((b-a)/h)):
    k1 = h*normal.normal(x_total[i],i*h)
    k2 = h*normal.normal(x_total[i]+k1/2,(i+0.5)*h)
    k3 = h*normal.normal(x_total[i]+k2,(i+1)*h)
    x_total[i+1] = x_total[i] + (k1 + 4*k2 + k3)/6
  return x_total

def RungeKutta44_SWE(x,dom_tem,lin, non_lin,normal,impli = False,tol = 10**(-5)):
  '''Essa função tem como entradas os parametros:
  x --- condição inicial
  a --- ponto inicial
  b --- ponto final
  h --- passo no tempo
  f --- função a ser aplicada'''
  ############################################################
  a = dom_tem.ti
  b = dom_tem.tf
  h = dom_tem.del_t
  ############################################################

  x_total = np.zeros([int((b-a)/h)+1,len(x)],dtype=complex)
  x_total[0] = np.copy(x)
  for i in range(int((b-a)/h)):
    k1 = h*normal.normal_SWE(x_total[i],i*h)
    k2 = h*normal.normal_SWE(x_total[i]+k1/2,(i+0.5)*h)
    k3 = h*normal.normal_SWE(x_total[i]+k2/2,(i+0.5)*h)
    k4 = h*normal.normal_SWE(x_total[i]+k3,(i+1)*h)
    x_total[i+1] = x_total[i] + (k1 + 2*k2 + 2*k3 + k4)/6
  return x_total