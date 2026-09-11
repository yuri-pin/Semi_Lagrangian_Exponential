## This document aims to define all the spatial and temporal domain
## of exponential integrators, semi-lagrangian and 
## semi-lagrangian exponential integrators

class Domain_temp:
  '''This class defines the temporal domain, where we are looking 
    for a del_t which divides the interval'''
  def __init__(self,ti=0,tf=1,del_t=0.01):
    self.ti = ti
    self.tf = tf
    self.del_t = del_t

class Domain_space:
  '''This class defines the spacial domain, where we are looking 
    for a del_h which divides the interval'''
  def __init__(self,a=0,b=1,del_h=0.01):
    self.a = a
    self.b = b
    self.del_h = del_h