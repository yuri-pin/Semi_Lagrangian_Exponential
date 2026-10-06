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
#########################################################################################################
#########################################################################################################
##################################  Exponential Integrators fundamental functions #######################
#########################################################################################################
#########################################################################################################

def ExpA(h,M):
  '''essa calculadora de exponencial de matriz só é válida para matriz diagonal
  sendo M --- diagonal da matriz que queremos encontrar a exponencial'''
  EA = np.exp(h*M)
  return EA



def phi_1(h, M):
    """
    Calcula phi_1(h,M) de forma vetorizada
    usando índices, sem for.
    """
    M = np.asarray(M, dtype=complex)
    n = M.shape[0]

    # vetor de índices
    indices = np.arange(n)

    # caso trivial
    if np.allclose(M, 0) or h == 0:
        return np.ones_like(M)

    # caso h muito pequeno
    if h <= 1e-8:
        return 1 + 0.5 * h * M

    # caso geral
    Exp = ExpA(h, M)     # se for só exp elementwise, pode usar np.exp(h*M)
    num = Exp - 1.0

    phi = np.empty_like(M)

    # índices válidos (M ≠ 0)
    mask = M != 0
    phi[indices[mask]] = num[indices[mask]] / (h * M[indices[mask]])

    # índices inválidos (M = 0) → limite = 1
    phi[indices[~mask]] = 1.0

    return phi

def phi_2(h, M):
    """
    Calcula phi_2(h,M) de forma vetorizada
    usando índices, sem for.
    """
    M = np.asarray(M, dtype=complex)
    n = M.shape[0]

    # vetor de índices
    indices = np.arange(n)

    # caso trivial
    if np.allclose(M, 0) or h == 0:
        return 0.5 * np.ones_like(M)

    # caso h muito pequeno
    if h <= 1e-8:
        return 0.5 + (1/6) * h * M

    # caso geral
    Exp = ExpA(h, M)   # se for só elementwise, pode ser np.exp(h*M)
    num = Exp - 1.0 - h*M

    phi = np.empty_like(M)

    # índices válidos (M ≠ 0)
    mask = M != 0
    phi[indices[mask]] = num[indices[mask]] / ((h*M[indices[mask]])**2)

    # índices inválidos (M = 0) → limite = 0.5
    phi[indices[~mask]] = 0.5

    return phi

#########################################################################################################
#########################################################################################################
##################################  Exponential Integrators fundamental functions #######################
#########################################################################################################
#########################################################################################################

def Euler_Exp_Explicito(x,dom_tem,lin, non_lin,normal,iter = True,tol = 10**(-5)):
  '''Essa função tem como entradas os parametros:
  x --- condição inicial
  a --- ponto inicial
  b --- ponto final
  h --- passo no tempo
  f --- função não linear
  M --- Parte linear'''
  ############################################################
  a = dom_tem.ti
  b = dom_tem.tf
  h = dom_tem.del_t
  ############################################################

  x_total = np.zeros([int((b-a)/h)+1,len(x)],dtype=complex)
  x_total[0] = np.copy(x)
  x_old = np.copy(x)
  x_new = np.zeros_like(x)
  #esta variavel exp_lin e phi_lin são para que o exponencial não seja calculado para toda iterada
  if (lin.ini == 0 and iter) or (lin.ini == 4 and iter):
    for i in range(int((b-a)/h)):
      x_total[i+1] = ExpA((i+1)*h,lin.lin(x=x))*x_total[0]
  else:
    exp_lin = ExpA(h,lin.lin(x=x))
    phi_lin = phi_1(h,lin.lin(x=x))


    for i in range(int((b-a)/h)):
      x_total[i+1] = exp_lin*x_total[i] + h*phi_lin*non_lin.nlin(x_total[i],i*h)
      x_old = np.copy(x_new)
  return x_total


def RK2_Exp_Explicito1(x,dom_tem,lin, non_lin,normal,impli = False,tol = 10**(-5)):
  '''
  This is the trapezoidal method
  Essa função tem como entradas os parametros:
  x --- condição inicial
  a --- ponto inicial
  b --- ponto final
  h --- passo no tempo
  f --- função não linear
  M --- Parte linear'''
  ############################################################
  a = dom_tem.ti
  b = dom_tem.tf
  h = dom_tem.del_t
  ############################################################

  c_2 = 1
  x_total = np.zeros([int((b-a)/h)+1,len(x)],dtype=complex)
  x_total[0] = np.copy(x)
  if lin.ini == 0 or (lin.ini == 4 ):
    for i in range(int((b-a)/h)):
      x_total[i+1] = ExpA((i+1)*h,lin.lin(x=x))*x_total[0]
  #esta variavel exp_lin e phi_lin são para que o exponencial não seja calculado para toda iterada
  else:
    exp_lin = ExpA(h,lin.lin(x=x))
    exp_lin2 = ExpA(c_2*h,lin.lin(x=x))#nesse programa consideramos c_2 = 1/2
    phi1_lin = phi_1(h,lin.lin(x=x))
    phi1_lin2 = phi_1(c_2*h,lin.lin(x=x))
    phi2_lin = phi_2(h,lin.lin(x=x))
    for i in range(int((b-a)/h)):
      k1 = exp_lin2*x_total[i] + h*c_2*phi1_lin2*non_lin.nlin(x_total[i],i*h)
      #faremos essa conta por partes
      lin = exp_lin*x_total[i]

      b1 = (phi1_lin - (1/c_2)*phi2_lin)*non_lin.nlin(x_total[i],i*h)

      b2 = (1/c_2)*phi2_lin*non_lin.nlin(k1,(i+c_2)*h)

      x_total[i+1] = lin + h*b1 + h*b2
  return x_total


def RK2_Exp_Explicito2(x,dom_tem,lin, non_lin,normal,impli = False,tol = 10**(-5)):
  '''
  This is the midpoint method
  Essa função tem como entradas os parametros:
  x --- condição inicial
  a --- ponto inicial
  b --- ponto final
  h --- passo no tempo
  f --- função não linear
  M --- Parte linear'''
  ############################################################
  a = dom_tem.ti
  b = dom_tem.tf
  h = dom_tem.del_t
  ############################################################


  c_2 = 1/2
  x_total = np.zeros([int((b-a)/h)+1,len(x)],dtype=complex)
  x_total[0] = np.copy(x)
  if lin.ini == 0 or (lin.ini == 4) :
    for i in range(int((b-a)/h)):
      x_total[i+1] = ExpA((i+1)*h,lin.lin(x=x))*x_total[0]
  #esta variavel exp_lin e phi_lin são para que o exponencial não seja calculado para toda iterada
  else:
    exp_lin = ExpA(h,lin.lin(x=x))
    exp_lin2 = ExpA(c_2*h,lin.lin(x=x))#nesse programa consideramos c_2 = 1/2
    phi1_lin = phi_1(h,lin.lin(x=x))
    phi1_lin2 = phi_1(c_2*h,lin.lin(x=x))
    phi2_lin = phi_2(h,lin.lin(x=x))
    for i in range(int((b-a)/h)):
      k1 = exp_lin2*x_total[i] + h*c_2*phi1_lin2*non_lin.nlin(x_total[i],i*h)
      #faremos essa conta por partes
      lin = exp_lin*x_total[i]

      b1 = (phi1_lin - (1/c_2)*phi2_lin)*non_lin.nlin(x_total[i],i*h)

      b2 = (1/c_2)*phi2_lin*non_lin.nlin(k1,(i+c_2)*h)

      x_total[i+1] = lin + h*b1 + h*b2
  return x_total


def IFEuler_Explicito(x,dom_tem,lin, non_lin,normal,impli = False,tol = 10**(-5)):
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


  x_total = np.zeros([int((b-a)/h)+1,len(x)])
  x_total[0] = np.copy(x)
  exp_lin = ExpA(h,lin.lin())
  for i in range(int((b-a)/h)):
    x_total[i+1] = np.dot(exp_lin, x_total[i] + h*non_lin.nlin(x_total[i],i*h))
  return x_total


def Lawson_2ordem_a(x,dom_tem,lin, non_lin,normal,impli = False,tol = 10**(-5)):
  '''Essa função tem como entradas os parametros:
  x --- condição inicial
  a --- ponto inicial
  b --- ponto final
  h --- passo no tempo
  f --- função a ser aplicada
  a_{12} = 1/2
  '''
  ############################################################
  a = dom_tem.ti
  b = dom_tem.tf
  h = dom_tem.del_t
  ############################################################


  x_total = np.zeros([int((b-a)/h)+1,len(x)])
  x_total[0] = np.copy(x)
  exp_lin = ExpA(h,lin.lin())
  exp_lin2 = ExpA(0.5*h,lin.lin())

  for i in range(int((b-a)/h)):

    k1 = non_lin.nlin(x_total[i],i*h)

    k2 = non_lin.nlin(np.dot(exp_lin2,x_total[i] + h*k1/2),(i+0.5)*h)

    x_total[i+1] = np.dot(exp_lin, x_total[i]) + h*np.dot(exp_lin2,k2)
  return x_total