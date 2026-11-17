import numpy as np
import domain as do

def anda(p, x):
    """
    Desloca cada elemento de x de acordo com p, com condição periódica.
    Se houver erro de índice, usa uma versão alternativa.
    """
    x = np.atleast_1d(x)
    p = np.atleast_1d(p)

    try:
        # Tenta executar a versão original que espera m > 1
        m, n = x.shape
        resultado = np.empty_like(x)
        for i in range(m):
            indices = (np.arange(n) - p[i].astype(int)) % n
            resultado[i] = x[i, indices]
        return resultado

    except Exception as e:
        # Se der erro (por exemplo, out of range), usa a versão simplificada
        n = len(x)
        indices = (np.arange(n) - p.astype(int)) % n
        return x[indices]


class interpol:
  '''Essa classe determina qual interpolação será feita entre os pontos
  determinados pelo donmínio'''
  def __init__(self,dom):
    self.dom = dom

  def P1(self,alp,p,x):
    """Função interpoladora de Lagrange linear, sendo que
    alp é o termo alpha que 0 < alpha < 1;
    x é o valor da função f nas posições."""
    x_ip   = anda(p,np.copy(x))
    #note que "q = p-1"
    x_iq = anda(1+p,np.copy(x))
    f = alp*x_iq + (1-alp)*x_ip
    return f

  def P2a(self,alp,p,x):
    """Função interpoladora de Lagrange quadrática, sendo que
    alp é o termo alpha que 0 < alpha < 1;
    x é o valor da função f nas posições."""
    #note que "ip = i-p"
    x_ip   = anda(p,np.copy(x))
    #note que "iq = i-p-1"
    x_iq = anda(1+p,np.copy(x))
    #note que "r = i-p-2"
    x_ir = anda(2+p,np.copy(x))
    f = (1-alp)*(2-alp)*0.5*x_ip + alp*(2-alp)*x_iq - alp*(1-alp)*0.5*x_ir
    return f

  def P2b(self,alp,p,x):
      """Função interpoladora de Lagrange quadrática, sendo que
      alp é o termo alpha que 0 < alpha < 1;
      x é o valor da função f nas posições."""
      #note que "ip = i-p"
      x_ip   = anda(0+p,np.copy(x))
      #note que "iq = i-p-1"
      x_iq = anda(1+p,np.copy(x))
      #note que "r = i-p+1"
      x_ir = anda(-1+p,np.copy(x))
      f = alp*(alp+1)*0.5*x_iq  -(alp-1)*(alp+1)*x_ip  + alp*(alp-1)*0.5*x_ir
      return f

  def P3(self,alp,p,x):
      """Função interpoladora de Lagrange cúbica, sendo que
      alp é o termo alpha que 0 < alpha < 1;
      x é o valor da função f nas posições."""
      #note que "is = i-p+1"
      x_is   = anda(-1+p,np.copy(x))
      #note que "ip = i-p"
      x_ip   = anda(p,np.copy(x))
      #note que "iq = i-p-1"
      x_iq = anda(1+p,np.copy(x))
      #note que "r = i-p-2"
      x_ir = anda(2+p,np.copy(x))
      f = ((1-alp)*(2-alp)*(1+alp)/2)*x_ip + ((alp)*(2-alp)*(1+alp)/2)*x_iq + ((alp)*(alp-1)*(1+alp)/6)*x_ir + ((-alp)*(2-alp)*(1-alp)/6)*x_is
      return f