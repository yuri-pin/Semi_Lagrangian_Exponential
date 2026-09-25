import numpy as np
import domain as do
import auxiliar as auxi


class linear_term_IE:
  ''' esse constroí o termo linear da equação, seja EDO ou EDP'''
  def __init__(self,ini = 1,spec = 0, dom_esp = do.Domain_space(), dom_tem = do.Domain_temp(),aux = auxi.aux()):
    self.ini  = ini
    '''ini = 0 ==> EDO linear
       ini = 1 ==> EDO Não linear Não stiff
       ini = 2 ==> EDO Não linear stiff
       ini = 3 ==> EDO Não linear stiff oscilatório
       ini = 4 ==> EDP advecção
       ini = 5 ==> EDP burgers'''
    self.spec = spec
    '''spec = 0 ==> não se usa método espectral
       spec = 1 ==> advecção velocidade constante
       spec = 2 ==> advecção velocidade variável
       spec = 3 ==> burgers'''
    self.dom_esp  = dom_esp
    self.dom_tem  = dom_tem
    self.aux = aux
def lin(self,x):
    if self.ini == 0:
      A = (np.array([2]))
      return A
    elif self.ini == 1:
      A = (np.array([2]))
      return A
    elif self.ini == 2:
      A = (np.array([-1000j]))
      return A
    elif self.ini == 3:
      A = (np.array([-1000]))
      return A
    elif self.ini == 4 and  (self.spec == 1 or self.spec == 2):
      N = int((len(x)//2))

      k = np.zeros(len(x), dtype = int)
      k[1:N+1] = np.linspace(1, N, N, endpoint = True)
      k[N+1:] = -np.flip(k[1:N+1])

      c = self.perf_vel(x = x)

      A = 2*np.pi*c*k*1j

      return A
    elif self.ini == 5 and self.spec == 3:
      N = int((len(x)//2))
      k = np.zeros(len(x), dtype = int)
      k[1:N+1] = np.linspace(1, N, N, endpoint = True)
      k[N+1:] = -np.flip(k[1:N+1])
      c = np.zeros_like(x)

      A = c*1j

      return A

    elif self.ini == 6 and self.spec == 3:
      M_line = len(x)
      N_line = int((M_line-1)/2)

      k = np.zeros(len(x), dtype = int)
      k[1:N_line+1] = np.linspace(1, N_line, N_line, endpoint = True)
      k[N_line+1:] = -np.flip(k[1:N_line+1])


      A = 0.01*(2*np.pi*k*1j)**2

      return A



class non_linear_term_IE:
  ''' esse constroí o termo linear da equação, seja EDO ou EDP'''
  def __init__(self,ini = 1,spec = 0, dom_esp = do.Domain_space(), dom_tem = do.Domain_temp(),aux = auxi.aux()):
    self.ini  = ini
    '''ini = 0 ==> EDO linear
       ini = 1 ==> EDO Não linear Não stiff
       ini = 2 ==> EDO Não linear stiff
       ini = 3 ==> EDO Não linear stiff oscilatório
       ini = 4 ==> EDP advecção
       ini = 5 ==> EDP burgers'''
    self.spec = spec
    '''spec = 0 ==> não se usa método espectral
       spec = 1 ==> advecção velocidade constante
       spec = 2 ==> advecção velocidade variável
       spec = 3 ==> burgers'''
    self.dom_esp  = dom_esp
    self.dom_tem  = dom_tem
    self.aux = aux

  def nlin(self,x,t):
    if self.ini == 0:
      y = 0
      return y
    elif self.ini == 1:
      y =  np.exp(2*t)
      return y
    elif self.ini == 2:
      y = np.exp(1j*t)
      return y
    elif self.ini ==3:
      y = -(-1000)*np.cos(t) - np.sin(t)
      return y
    elif self.ini == 4 and (self.spec == 1 or self.spec == 2):
      N = len(x)
      A = np.zeros(N,dtype=complex)

      return A
    elif self.ini == 5 and self.spec == 3:
      M_line = len(x)
      N_line = int((M_line-1)/2)

      k = np.zeros(len(x), dtype = int)
      k[1:N_line+1] = np.linspace(1, N_line, N_line, endpoint = True)
      k[N_line+1:] = -np.flip(k[1:N_line+1])

      x_redef,k_redef = self.aux.redef_f(f = np.copy(x), k = k,N = N_line) # redefinir no espaço espectral


      x_deriv = 2*np.pi*k_redef*1j*x_redef # definimos a derivada no espaço espectral

      w_real = -self.aux.IFFT(x_redef)*self.aux.IFFT(x_deriv)
      A,k = self.aux.FFT(w_real)

      return A

    elif self.ini == 6 and self.spec == 3:
      M_line = len(x)
      N_line = int((M_line-1)/2)

      k = np.zeros(len(x), dtype = int)
      k[1:N_line+1] = np.linspace(1, N_line, N_line, endpoint = True)
      k[N_line+1:] = -np.flip(k[1:N_line+1])

      x_redef,k_redef = self.aux.redef_f(f = np.copy(x), k = k,N = N_line) # redefinir no espaço espectral

      x_new = np.arange(self.dom_esp.a,self.dom_esp.b,1/len(x))

      x_deriv = 2*np.pi*k_redef*1j*x_redef # definimos a derivada no espaço espectral

      w_real = -self.aux.IFFT(x_redef)*self.aux.IFFT(x_deriv) - 1*self.aux.IFFT(x)

      A,k = self.aux.FFT(w_real)

      return A


class normal_term_IE:
  ''' esse constroí o termo linear da equação, seja EDO ou EDP'''
  def __init__(self,ini = 1,spec = 0, dom_esp = do.Domain_space(), dom_tem = do.Domain_temp(),aux = auxi.aux()):
    self.ini  = ini
    '''ini = 0 ==> EDO linear
       ini = 1 ==> EDO Não linear Não stiff
       ini = 2 ==> EDO Não linear stiff
       ini = 3 ==> EDO Não linear stiff oscilatório
       ini = 4 ==> EDP advecção
       ini = 5 ==> EDP burgers'''
    self.spec = spec
    '''spec = 0 ==> não se usa método espectral
       spec = 1 ==> advecção velocidade constante
       spec = 2 ==> advecção velocidade variável
       spec = 3 ==> burgers'''
    self.dom_esp  = dom_esp
    self.dom_tem  = dom_tem
    self.aux = aux

  def normal(self,x:np.ndarray,t:float):
    '''Essa função tem como entradas os parametros:
    x --- a posição no tempo t
    t --- tempo'''
    if self.ini == 0: #nesse caso vamos testar que para o caso puramente linear Euler_Exp_explicito é exato
      A = np.diag(np.array([2]))
      y = np.dot(A,x)
      return y
    elif self.ini == 1: #x(t) = exp(2t)*t + 2*exp(2t)
      A = np.diag(np.array([2]))
      y = np.dot(A,x) + np.exp(2*t)
      return y
    elif self.ini == 2:  #x(t) = -(100/10001)cos(10t)-(1/10001)sin(10t) + (10101/10001)*exp(t)
      A = np.diag(np.array([-1000j]))
      y = np.dot(A,x) + np.exp(1j*t)
      return y
    elif self.ini == 3:  #x(t) = cos(t) + exp(-1000t)
      A = np.diag(np.array([-1000]))
      y = np.dot(A,x) - (-1000)*np.cos(t) - np.sin(t)
      return y
    elif self.ini == 4 and (self.spec == 1 or self.spec == 2):
      A = self.test_exp_lin(x = x)
      y = A*x + self.test_exp_nlin(x,t)
      return y
    elif self.ini == 5 and self.spec == 3:
      A = self.test_exp_lin(x = x)
      y = A*x + self.test_exp_nlin(x,t)
      return y
    elif self.ini == 6 and self.spec == 3:
      A = self.test_exp_lin(x = x)
      y = A*x + self.test_exp_nlin(x,t)
      return y


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################


class linear_term_SE:
  ''' esse constroí o termo linear da equação, seja EDO ou EDP'''
  def __init__(self,ini = 1,spec = 0, dom_esp = do.Domain_space(), dom_tem = do.Domain_temp(),aux = auxi.aux()):
    self.ini  = ini
    '''ini = 0 ==> EDO linear
       ini = 1 ==> EDO Não linear Não stiff
       ini = 2 ==> EDO Não linear stiff
       ini = 3 ==> EDO Não linear stiff oscilatório
       ini = 4 ==> EDP advecção
       ini = 5 ==> EDP burgers'''
    self.spec = spec
    '''spec = 0 ==> não se usa método espectral
       spec = 1 ==> advecção velocidade constante
       spec = 2 ==> advecção velocidade variável
       spec = 3 ==> burgers'''
    self.dom_esp  = dom_esp
    self.dom_tem  = dom_tem
    self.aux = aux

  def lin(self,x):
    '''Esse x é o espaço físico discretizado'''
    if self.ini == 0 and self.spec == 0:
      A = (np.array([-1]))
      return A
    elif self.ini == 1 and self.spec == 0:
      A = (np.array([0]))
      return A
    elif self.ini == 2 and  self.spec == 0:
      A = np.zeros_like(x)
      return A
    elif self.ini == 2 and  self.spec == 1:
      M_line = len(x)
      N_line = int((M_line-1)/2)

      k = np.zeros(len(x), dtype = int)
      k[1:N_line+1] = np.linspace(1, N_line, N_line, endpoint = True)
      k[N_line+1:] = -np.flip(k[1:N_line+1])


      A = 0.01*(2*np.pi*k*1j)**2
      return A
    elif self.ini == 3 and self.spec ==0:
      A = np.array([0])
      return A
    elif self.ini == 4 and self.spec == 0:
      A = np.sin(2*np.pi*x/(self.dom_esp.b - self.dom_esp.a))
      return A
    elif self.ini == 5 and self.spec == 0:
      A = np.sin(2*np.pi*x/(self.dom_esp.b - self.dom_esp.a))*0.5
      return A
    elif self.ini == 6 and self.spec == 1:
      c = np.zeros_like(x)
      A = c*1j
      return A
    elif self.ini == 6 and self.spec ==0:
      A = np.array([0])
      return A
    elif self.ini ==7 and self.spec ==0:
      A = -np.sin(2*np.pi*x/(self.dom_esp.b - self.dom_esp.a))
      return A
    elif self.ini == 7 and self.spec ==1:
      c = np.zeros_like(x)
      A = c*1j
      return A
    elif self.ini == 8 and self.spec ==0:
      A = np.array([-np.sqrt(2)])
      return A
    elif self.ini == 8 and self.spec ==1:
      c = np.zeros_like(x)
      A = c*1j
      return A
    elif self.ini == 9 and  self.spec == 1:
      ###Não sei o que fazer aqui!!!!
      M_line = len(x)
      N_line = int((M_line-1)/2)

      k = np.zeros(len(x), dtype = int)
      k[1:N_line+1] = np.linspace(1, N_line, N_line, endpoint = True)
      k[N_line+1:] = -np.flip(k[1:N_line+1])


      A = 0.01*(2*np.pi*k*1j)**2
      return A
    elif self.ini == 10 and self.spec ==1:
      M_line = len(x)
      N_line = int((M_line-1)/2)

      k = np.zeros(len(x), dtype = int)
      k[1:N_line+1] = np.linspace(1, N_line, N_line, endpoint = True)
      k[N_line+1:] = -np.flip(k[1:N_line+1])


      A = 0.01*(2*np.pi*k*1j)**2
      return A





    elif self.ini == 11 and  self.spec == 1:
      ###Não sei o que fazer aqui!!!!
      M_line = len(x)
      N_line = int((M_line-1)/2)

      k = np.zeros(len(x), dtype = int)
      k[1:N_line+1] = np.linspace(1, N_line, N_line, endpoint = True)
      k[N_line+1:] = -np.flip(k[1:N_line+1])


      A = 0.01*(2*np.pi*k*1j)**2
      return A
    elif self.ini == 12 and self.spec ==1:
      M_line = len(x)
      N_line = int((M_line-1)/2)

      k = np.zeros(len(x), dtype = int)
      k[1:N_line+1] = np.linspace(1, N_line, N_line, endpoint = True)
      k[N_line+1:] = -np.flip(k[1:N_line+1])


      A = 0.01*(2*np.pi*k*1j)**2
      return A

  def L_SWE(self,x,h,func,f = 10**(0),alt = 1, g = 10):
    if self.ini == 0 and self.spec ==0:
      M_line = len(x)
      N_line = int((M_line-1)/2)

      k = np.zeros(len(x), dtype = int)
      k[1:N_line+1] = np.linspace(1, N_line, N_line, endpoint = True)
      k[N_line+1:] = -np.flip(k[1:N_line+1])

      A = self.aux.A_all(k, h, func, f, alt, g)
      return A
    elif self.ini == 0 and self.spec ==1:
      M_line = len(x)
      N_line = int((M_line-1)/2)

      k = np.zeros(len(x), dtype = int)
      k[1:N_line+1] = np.linspace(1, N_line, N_line, endpoint = True)
      k[N_line+1:] = -np.flip(k[1:N_line+1])

      A = self.aux.A_all(k, h, func, f, alt, g)
      return A

class non_linear_term_SE:
  ''' esse constroí o termo não linear da equação, seja EDO ou EDP'''
  def __init__(self,ini = 1,spec = 0, dom_esp = do.Domain_space(), dom_tem = do.Domain_temp(),aux = auxi.aux()):
    self.ini  = ini
    '''ini = 0 ==> EDO linear
       ini = 1 ==> EDO Não linear Não stiff
       ini = 2 ==> EDO Não linear stiff
       ini = 3 ==> EDO Não linear stiff oscilatório
       ini = 4 ==> EDP advecção
       ini = 5 ==> EDP burgers'''
    self.spec = spec
    '''spec = 0 ==> não se usa método espectral
       spec = 1 ==> advecção velocidade constante
       spec = 2 ==> advecção velocidade variável
       spec = 3 ==> burgers'''
    self.dom_esp  = dom_esp
    self.dom_tem  = dom_tem
    self.aux = aux

  def nlin(self,x,t):
    if self.ini == 0 and self.spec == 0:
      y = 0
      return y
    elif self.ini == 1 and self.spec == 0:
      y =  -1*x
      return y
    elif self.ini == 2 and self.spec == 0:
      M_line = len(x)
      N_line = int((M_line-1)/2)

      k = np.zeros(len(x), dtype = int)
      k[1:N_line+1] = np.linspace(1, N_line, N_line, endpoint = True)
      k[N_line+1:] = -np.flip(k[1:N_line+1])

      A = (0.01*(2*np.pi*k*1j)**2)*x - np.sqrt(2)*2*np.pi*k*1j*x -1*x
      return A
    elif self.ini ==2 and self.spec ==1:
      A = -1*x
      return A
    elif self.ini ==3 and self.spec ==0:
      y = 0
      return y
    elif self.ini == 4 and self.spec == 0:
      y = 0
      return y
    elif self.ini == 5 and self.spec == 0:
      x_space = np.arange(self.dom_esp.a,self.dom_esp.b,1/len(x))
      A = np.sin(2*np.pi*x_space/(self.dom_esp.b - self.dom_esp.a))*0.5*x
      return A
    elif self.ini == 6 and self.spec == 1:
      M_line = len(x)
      N_line = int((M_line-1)/2)

      k = np.zeros(len(x), dtype = int)
      k[1:N_line+1] = np.linspace(1, N_line, N_line, endpoint = True)
      k[N_line+1:] = -np.flip(k[1:N_line+1])

      x_redef,k_redef = self.aux.redef_f(f = np.copy(x), k = k,N = N_line) # redefinir no espaço espectral

      x_deriv = 2*np.pi*k_redef*1j*x_redef # definimos a derivada no espaço espectral

      w_real = -self.aux.IFFT(x_redef)*self.aux.IFFT(x_deriv)
      A,k = self.aux.FFT(w_real)

      return A
    elif self.ini == 6 and self.spec ==0:
      A = np.array([0])
      return A
    elif self.ini == 7 and self.spec ==0:
      A = np.array([0])
      return A
    elif self.ini == 7 and self.spec ==1:
      M_line = len(x)
      N_line = int((M_line-1)/2)


      k = np.zeros(len(x), dtype = int)
      k[1:N_line+1] = np.linspace(1, N_line, N_line, endpoint = True)
      k[N_line+1:] = -np.flip(k[1:N_line+1])

      x_redef,k_redef = self.aux.redef_f(f = np.copy(x), k = k,N = N_line) # redefinir no espaço espectral

      x_deriv = 2*np.pi*k_redef*1j*x_redef # definimos a derivada no espaço espectral

      x_new = np.arange(self.dom_esp.a,self.dom_esp.b,1/len(x))

      w_real = -self.aux.IFFT(x_redef)*self.aux.IFFT(x_deriv) +(- np.sin(2*np.pi*x_new/(self.dom_esp.b - self.dom_esp.a)))*IFFT(x)
      A,k = self.aux.FFT(w_real)
      return A
    elif self.ini == 8 and self.spec ==0:
      A = np.array([0])
      return A
    elif self.ini == 8 and self.spec ==1:
      M_line = len(x)
      N_line = int((M_line-1)/2)


      k = np.zeros(len(x), dtype = int)
      k[1:N_line+1] = np.linspace(1, N_line, N_line, endpoint = True)
      k[N_line+1:] = -np.flip(k[1:N_line+1])

      x_redef,k_redef = self.aux.redef_f(f = np.copy(x), k = k,N = N_line) # redefinir no espaço espectral

      x_deriv = 2*np.pi*k_redef*1j*x_redef # definimos a derivada no espaço espectral

      x_new = np.arange(self.dom_esp.a,self.dom_esp.b,1/len(x))

      w_real = -self.aux.IFFT(x_redef)*self.aux.IFFT(x_deriv) - np.sqrt(2)*self.aux.IFFT(x)
      A,k = self.aux.FFT(w_real)
      return A



    elif self.ini == 9 and self.spec ==1:
      M_line = len(x)
      N_line = int((M_line-1)/2)

      k = np.zeros(len(x), dtype = int)
      k[1:N_line+1] = np.linspace(1, N_line, N_line, endpoint = True)
      k[N_line+1:] = -np.flip(k[1:N_line+1])

      x_redef,k_redef = self.aux.redef_f(f = np.copy(x), k = k,N = N_line) # redefinir no espaço espectral

      x_deriv = 2*np.pi*k_redef*1j*x_redef  # definimos a derivada no espaço espectral



      w_real = -self.aux.IFFT(x_redef)*self.aux.IFFT(x_deriv) -self.aux.IFFT((x))
      A,k = self.aux.FFT(w_real)
      return A

    elif self.ini == 10 and self.spec ==1:
      A = -x
      return A


     ################ aqui o termo não linear será u^2
    elif self.ini == 11 and self.spec ==1:
      M_line = len(x)
      N_line = int((M_line-1)/2)

      k = np.zeros(len(x), dtype = int)
      k[1:N_line+1] = np.linspace(1, N_line, N_line, endpoint = True)
      k[N_line+1:] = -np.flip(k[1:N_line+1])

      x_redef,k_redef = self.aux.redef_f(f = np.copy(x), k = k,N = N_line) # redefinir no espaço espectral

      x_deriv = 2*np.pi*k_redef*1j*x_redef  # definimos a derivada no espaço espectral



      w_real = -self.aux.IFFT(x_redef)*self.aux.IFFT(x_deriv)-1*self.aux.IFFT((x))**2
      A,k = self.aux.FFT(w_real)
      return A

    elif self.ini == 12 and self.spec ==1:
      A = -(x**2)
      return A


  def N_SWE(self,x,t):
    if self.ini == 0 and self.spec ==0:
      u,v,h = auxi.desmontar_U(x)
      n = np.zeros_like(u, dtype = complex)

      u_tilde,k_tilde = auxi.FFT(u)
      M_line = len(u)
      N_line = int((M_line-1)/2)

      u_redef,k_redef = auxi.redef_f(f = np.copy(u_tilde), k = k_tilde,N = N_line) # redefinir no espaço espectral
      u_deriv = 2*np.pi*k_redef*1j*u_redef  # definimos a derivada no espaço espectral

      w_real = h*auxi.IFFT(u_deriv)

      N = -1*auxi.montar_U(n,n,w_real)


      return N
    elif self.ini == 0 and self.spec ==1:
      u,v,h = auxi.desmontar_U(x) # desmonta no espectral
      n = np.zeros_like(u, dtype = complex)

      _,k_tilde = auxi.FFT(u)
      M_line = len(u)
      N_line = int((M_line-1)/2)

      u_redef,k_redef = auxi.redef_f(f = np.copy(u), k = k_tilde,N = N_line) # redefinir no espaço espectral
      v_redef,_ = auxi.redef_f(f = np.copy(v), k = k_tilde,N = N_line) # redefinir no espaço espectral
      h_redef,_ = auxi.redef_f(f = np.copy(h), k = k_tilde,N = N_line) # redefinir no espaço espectral

      u_deriv = 2*np.pi*k_redef*1j*u_redef  # definimos a derivada no espaço espectral
      v_deriv = 2*np.pi*k_redef*1j*v_redef  # definimos a derivada no espaço espectral
      h_deriv = 2*np.pi*k_redef*1j*h_redef  # definimos a derivada no espaço espectral



      u_varia = auxi.IFFT(u_redef)*auxi.IFFT(u_deriv)
      v_varia = auxi.IFFT(u_redef)*auxi.IFFT(v_deriv)

      h_varia = auxi.IFFT(h_redef)*auxi.IFFT(u_deriv) + auxi.IFFT(u_redef)*auxi.IFFT(h_deriv) # no espaço físico


      u_tilde,_ = auxi.FFT(u_varia)
      v_tilde,_ = auxi.FFT(v_varia)
      h_tilde,_ = auxi.FFT(h_varia)

      N = -1*auxi.montar_U(u_tilde,v_tilde,h_tilde)


      return N

class normal_term_SE:
  '''essa classe compõe o termo linear e o termo não linear para os métodos convencionais'''
  def __init__(self,ini = 1,spec = 0, dom_esp = do.Domain_space(), dom_tem = do.Domain_temp(), aux = auxi.aux()):
    self.ini  = ini
    self.spec = spec
    self.dom_esp  = dom_esp
    self.dom_tem  = dom_tem
    self.aux = aux
    self.lin = linear_term_SE(ini=ini, spec=spec,
                               dom_esp=dom_esp, dom_tem=dom_tem, aux=aux)
    self.non_lin = non_linear_term_SE(ini=ini, spec=spec,
                                   dom_esp=dom_esp, dom_tem=dom_tem, aux=aux)

  def normal(self,x:np.ndarray,t:float):
    '''Essa função tem como entradas os parametros:
    x --- a posição no tempo t
    t --- tempo'''
    A = self.lin.lin(x = x)
    y = A*x + self.non_lin.nlin(x,t)
    return y

  def normal_SWE(self,x,t,func = auxi.aux.transforma_em_matriz,f = 10**(0),alt = 1, g = 10):
    h = self.dom_esp.del_h
    A = self.lin.L_SWE(x,h,func,f = 10**(0),alt = 1, g = 10)
    u_L,v_L,h_L = auxi.desmontar_U(auxi.prod_lin_esp_SWE(x,A))
    u_tilde, _ = auxi.FFT(u_L)
    v_tilde, _ = auxi.FFT(v_L)
    h_tilde, _ = auxi.FFT(h_L)
    y_linear = auxi.montar_U(u_tilde, v_tilde, h_tilde)


    y = y_linear + self.non_lin.N_SWE(x,t)
    return y