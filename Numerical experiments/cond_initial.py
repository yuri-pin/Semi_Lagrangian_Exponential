import numpy as np
import domain 

class cond_initial:
  '''esse constroí a condição inicial de cada problema'''
  def __init__(self,cond = 0, dom_esp = domain.Domain_space()):
    self.cond  = cond
    '''ini = 0 ==> EDO linear
       ini = 1 ==> EDO Não linear Não stiff
       ini = 2 ==> EDO Não linear stiff
       ini = 3 ==> EDO Não linear stiff oscilatório
       ini = 4 ==> EDP advecção
       ini = 5 ==> EDP burgers'''
    self.dom_esp  = dom_esp


  def cond_ini_IE(self, x =0):
    '''This function will generate the initial conditions for the 
    exponential integrators numerical experiments
       cond = 0 ==> EDO linear
       cond = 1 ==> EDO Não linear Não stiff
       cond = 2 ==> EDO Não linear stiff
       cond = 3 ==> EDO Não linear stiff oscilatório
       cond = 4 ==> EDP advecção
       cond = 5 ==> EDP burgers
    '''
    return 


  def cond_ini_SL(self,x=0):
    '''This function will generate the initial condition for the 
    semi-Lagrangian numerical experiments'''

    return 
  def cond_ini_SE(self,x = 0):
    '''This function will generate the initial condition for the 
    semi-Lagrangian exponential integrators numerical experiments
       cond = 0 ==> sinusoidal
       cond = 1 ==> gaussian
       cond = 2 ==> triangular
       cond = 3 ==> square
    '''
    if self.cond == 0:
      '''Essa função tem como condição inicial a função seno
      As entradas dessa função são:
        x - pontos determinados entre a e b'''
      f = np.sin(2*np.pi*x/(self.dom_esp.b - self.dom_esp.a))
      return f
    elif self.cond == 1:
      '''Essa função tem como condição inicial a função gaussiana com centro em
      0.5 e variância de 0.1
      As entradas dessa função são:
        x - pontos determinados entre a e b'''
      f = np.exp(-0.5*(x-(0.5*self.dom_esp.a + 0.5*self.dom_esp.b))**2/0.01)
      return f
    elif self.cond == 2:
      '''Essa função tem como condição inicial a função quadrada cuja largura é
      de 0.4
      As entradas dessa função são:
        x - pontos determinados entre a e b'''
      m = len(x)
      f = np.zeros(m)
      for i in range (m):
        if x[i] >= 0.3 and x[i] <= 0.15 +np.sqrt(2)/4:
          f[i] = x[i] - 0.3
        elif x[i] >= 0.15 +np.sqrt(2)/4 and x[i] <= np.sqrt(2)/2:
          f[i] = np.sqrt(2)/2 - x[i]
        else:
          f[i] = 0
      return f
    elif self.cond == 3:
      '''Essa função tem como condição inicial a função triangular
      As entradas dessa função são:
        x - pontos determinados entre a e b'''
      m = len(x)
      f = np.zeros(m)
      for i in range (m):
        if x[i] >= 0.3 and x[i] <= 0.700001:
          f[i] = 1
        else:
          f[i] = 0
      return f

  def cond_ini_SWE(self,x,f = 10**(0),g = 10, alt = 1):
    '''This function will generate the initial condition for the 
        shallow water equation numerical experiments, where its
        initial conditions corresponds to the geostrophic mode.
        '''
    u = np.zeros_like(x)
    v = -(g/f)*(2*np.pi/(self.dom_esp.b - self.dom_esp.a))*np.sin(2*np.pi*x/(self.dom_esp.b - self.dom_esp.a))
    h = np.cos(2*np.pi*x/(self.dom_esp.b - self.dom_esp.a))

    U_init = montar_U(u,v,h)


    return U_init