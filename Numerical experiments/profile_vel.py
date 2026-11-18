import numpy as np

class perf_vel:
    '''Essa classe determina qual será o perfil de velocidade
    que será usado no problema, sendo esse perfil constante ou não.'''

    def __init__(self,dom_esp,dom_temp, vel = 0):
        self.velo = vel
        self.dom_esp = dom_esp
        self.dom_temp = dom_temp

    def profile(self, x, t=0):
        '''
        Define o perfil de velocidade utilizado no problema.

        Parâmetros:
            x   - posições dentro do domínio entre a e b
            t   - tempo
            vel - escolha do perfil de velocidade:

                  0: velocidade constante sqrt(2)
                  1: velocidade constante 1
                  2: velocidade variável no espaço
                  3: velocidade variável no espaço e tempo (exponencial)
                  4: velocidade variável no espaço e tempo (senoidal)
                  5: velocidade variável no tempo com sqrt(1)
                  6: velocidade variável no tempo com sqrt(2)

        Retorna:
            c   - velocidade de advecção
            p   - parte inteira do número de Courant
            alp - parte fracionária do número de Courant
        '''

        if self.velo == 0:
            c = np.sqrt(2) * np.ones(len(x))

        elif self.velo == 1:
            c = np.ones(len(x))

        elif self.velo== 2:
            c = 10 * np.sin(x * np.pi / (self.dom.b - self.dom.a)) + 1

        elif self.velo == 3:
            c = 10 * np.sin(x * np.pi / (self.dom.b - self.dom.a)) * np.exp(-t / 10) + 1

        elif self.velo == 4:
            c = 10 * np.sin(x * np.pi / (self.dom.b - self.dom.a)) * np.sin(
                2 * np.pi * t / (self.dom.tf - self.dom.ti)
            )

        elif self.velo == 5:
            c = (10 * np.sin(2 * np.pi * t / (self.dom.tf - self.dom.ti)) + np.sqrt(1)) * np.ones_like(x)

        elif self.velo == 6:
            c = (10 * np.sin(2 * np.pi * t / (self.dom.tf - self.dom.ti)) + np.sqrt(2)) * np.ones_like(x)


        p = (c * self.dom_temp.del_t / self.dom_esp.del_h) // 1
        alp = (c * self.dom_temp.del_t / self.dom_esp.del_h) % 1

        return np.array([c, p, alp])
