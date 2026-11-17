import numpy as np

class perf_vel:
    '''Essa classe determina qual será o perfil de velocidade
    que será usado no problema, sendo esse perfil constante ou não.'''

    def __init__(self, dom, cond, ini=1):
        self.ini = ini
        self.dom = dom

    def profile(self, x, t=0, vel=0):
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

        if vel == 0:
            c = np.sqrt(2) * np.ones(len(x))

        elif vel == 1:
            c = np.ones(len(x))

        elif vel == 2:
            c = 10 * np.sin(x * np.pi / (self.dom.b - self.dom.a)) + 1

        elif vel == 3:
            c = 10 * np.sin(x * np.pi / (self.dom.b - self.dom.a)) * np.exp(-t / 10) + 1

        elif vel == 4:
            c = 10 * np.sin(x * np.pi / (self.dom.b - self.dom.a)) * np.sin(
                2 * np.pi * t / (self.dom.tf - self.dom.ti)
            )

        elif vel == 5:
            c = (10 * np.sin(2 * np.pi * t / (self.dom.tf - self.dom.ti)) + np.sqrt(1)) * np.ones_like(x)

        elif vel == 6:
            c = (10 * np.sin(2 * np.pi * t / (self.dom.tf - self.dom.ti)) + np.sqrt(2)) * np.ones_like(x)


        p = (c * self.dom.del_t / self.dom.del_h) // 1
        alp = (c * self.dom.del_t / self.dom.del_h) % 1

        return np.array([c, p, alp])
