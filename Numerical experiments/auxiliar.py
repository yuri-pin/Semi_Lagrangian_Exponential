import numpy as np
import domain as do

####################################################################
# This set of functions aim to help de construction of numerical 
# methods, and some technical support for the numerical methods 
# mainly the spectral method, and the semi-Lagrangian exponential 
# methods for the SWE.

class aux:
    def __init__(self,ini = 1,spec = 0, dom_esp = do.Domain_space(), dom_tem = do.Domain_temp()):
        self.ini  = ini
        self.spec = spec
        self.dom_esp  = dom_esp
        self.dom_tem  = dom_tem


    def reestrut(self,n,x_i,x):
        """
        This function helps us to compare two different solutions with multiple sizes.
        that is, the vector x_i is smaller then the x, however the positions associated 
        with x_i are contained in x
        
        Parameters:
        --------------------------------
        n   : Ratio of sizes between x and x_i
        x_i : Smaller vector contained in the x
        x   : Larger vector that we want to compare with x_i 
        """
        k = len(x_i)
        indices = (n)*np.arange(k)
        x_new = x[indices]
        return x_new

    def erro2(self, x,y,dx):
        """
        This functions evaluate the error between two functions in the spatial domain 
        in the norm L^2. It is focused for the PDE comparison.

        Parametes:
        -------------------------------------
        x,y : Functions that we want to compare
        dx  : Spatial discretization of the problem
        """
        err = np.sqrt(np.sum((x-y)**2)*dx)
        return err

    def perf_vel(self,x = 0):
        if self.ini == 2 and self.spec == 1:
            c = -np.ones_like(x)
        return c


    #############################################
    #descobrir o que isso faz 
    #############################################

    def transforma_em_matriz(self,h,M):
        A = (M)
        return A

        #aqui dfiniremos a matriz Q para cada frequencia


    def Q_k(self,k, f, h, g):
        '''
        This functions is a spectral representation of a invertible matrix 
        given by the diagonalization process of spectral Shallow Water Equations

        Parameters:
        -----------------------------------
        k : wavenumber
        f : Coriolis term
        h : mean height of fluid
        g : gravity
        '''
        k = 2*np.pi*k
        # Caso especial k = 0
        if k == 0:
            return np.array([
                [0, -1j, 1j],
                [0, 1,   1 ],
                [1, 0,   0 ]
            ], dtype=complex)

        # Caso geral
        raiz = np.sqrt(f**2 + g*h*k**2)

        M = np.array([
            [0,             -raiz/(h*k),    raiz/(h*k)],
            [1j*g*k/f,       -1j*f/(h*k),     -1j*f/(h*k)],
            [1,               1,              1]
        ], dtype=complex)

        return M



    def Q_minus_k(self,k, f, h, g):
        '''
        This functions is a spectral representation of a inverse of Q_k matrix 
        given by the diagonalization process of spectral Shallow Water Equations

        Parameters:
        -----------------------------------
        k : wavenumber
        f : Coriolis term
        h : mean height of fluid
        g : gravity
        '''
        k = 2*np.pi*k
        # Caso especial k = 0
        if k == 0:
            return np.array([
                [0,     0,   1],
                [1j/2, 1/2, 0],
                [- 1j/2, 1/2, 0]
            ], dtype=complex)

        # Caso geral
        raiz = np.sqrt(f**2 + g*h*k**2)

        M = np.array([
            [0,                      -1j*f*h*k/(f**2 + g*h*k**2),     f**2/(f**2 + g*h*k**2)],

            [-h*k/(2*raiz),         1j*f*h*k/(2*f**2 + 2*g*h*k**2), g*h*k**2/(2*f**2 + 2*g*h*k**2)],

            [h*k/(2*raiz),          1j*f*h*k/(2*f**2 + 2*g*h*k**2),  g*h*k**2/(2*f**2 + 2*g*h*k**2)]
        ], dtype=complex)

        return M


    def Lambda_k(self,k, f, h, g):
        '''
        This functions is a spectral representation of a diagonal matrix 
        given by the diagonalization process of spectral Shallow Water Equation

        Parameters:
        -----------------------------------
        k : wavenumber
        f : Coriolis term
        h : mean height of fluid
        g : gravity
        '''
        k = 2*np.pi*k
        raiz = np.sqrt(f**2 + (g*h)*k**2)

        M = np.zeros(3, dtype=complex)

        M[0] = 0
        M[1] = 1j*raiz
        M[2] = -1j*raiz

        return M

    def A_all(self,k, h, func, f = 1, alt = 1, g = 1):
        """
        Retorn a matrix composed as

        A(k_i) = Q_k @ diag(func(h, Lambda_k)) @ Q_minus_k

        for each k_i.

        Parameters:
        ----------
        k    : array which contains the k_i
        f    : Coriolis term 
        alt  : mean height
        g    : gravity
        h    : time step
        func : função (phi_1, phi_2, ExpA, etc.)
        """

        return np.array([
            self.Q_k(ki, f, alt, g)
            @ np.diag(func(h, self.Lambda_k(ki, f, alt, g)))
            @ self.Q_minus_k(ki, f, alt, g)
            for ki in k])

    def montar_U(self,u, v, h):
        """
        This function compose 3 vectors in new one greater and return

        U = (u1, v1, h1, u2, v2, h2, ..., un, vn, hn)

        Parameters:
        -------------------------------------
        u : velocity field in x
        v : velocity field in y
        h : height of the fluid
        """

        u = np.asarray(u)
        v = np.asarray(v)
        h = np.asarray(h)

        # Verifica se possuem o mesmo tamanho
        if not (len(u) == len(v) == len(h)):
            raise ValueError("u, v e h devem ter o mesmo comprimento.")

        # Intercala os vetores
        U = np.column_stack((u, v, h)).ravel()

        return U

    def desmontar_U(self,U):
        """
        As a entry receive:

        U = (u1,v1,h1,u2,v2,h2,...,un,vn,hn)

        and return the vectors u, v e h separately.

        Parameters:
        ---------------------------------------------
        U: composition of vectors fields and height in the shallow water equations
        """

        U = np.asarray(U)

        # Verifica se o tamanho é múltiplo de 3
        if len(U) % 3 != 0:
            raise ValueError("O tamanho de U deve ser múltiplo de 3.")

        # Reorganiza em linhas [u_i, v_i, h_i]
        M = U.reshape(-1, 3)

        u = M[:, 0]
        v = M[:, 1]
        h = M[:, 2]

        return u, v, h


    ###########################################
    # Corrigir esses comentários
    ###########################################
    def prod_lin_esp_SWE(self,U, A):
        """
        Aplica o operador espectral A(k_i) ao vetor físico

            U = (u1,v1,h1,u2,v2,h2,...)

        onde:
            A[i] é uma matriz 3x3 associada ao modo k_i.

        Pipeline:
            U físico
            -> separa u,v,h
            -> FFT
            -> aplica A(k_i) em cada modo
            -> IFFT
            -> remonta U físico

        Parâmetros
        ----------
        U : array (3N,)
            Vetor no espaço físico.

        A : array (N,3,3)
            Matrizes espectrais retornadas por A_all.

        Retorna
        -------
        U_new : array (3N,)
            Vetor no espaço físico após aplicação do operador.
        """

        # ---------------------------------------------------
        # 1) separa variáveis físicas
        # ---------------------------------------------------

        u, v, h = self.desmontar_U(U)

        # ---------------------------------------------------
        # 2) FFT de cada variável
        # ---------------------------------------------------

        u_tilde, _ = self.FFT(np.copy(u))
        v_tilde, _ = self.FFT(np.copy(v))
        h_tilde, _ = self.FFT(np.copy(h))

        # ---------------------------------------------------
        # 3) monta vetor espectral intercalado
        # ---------------------------------------------------

        U_tilde = self.montar_U(u_tilde, v_tilde, h_tilde)

        # número de modos
        N = len(u_tilde)

        # ---------------------------------------------------
        # 4) aplica A(k_i) em cada modo
        # ---------------------------------------------------

        U_tilde_new = np.zeros_like(U_tilde, dtype=complex)

        for i in range(N):

            # bloco espectral do modo i
            Xi = U_tilde[3*i : 3*i + 3]

            # aplica matriz 3x3
            Yi = A[i] @ Xi

            # salva
            U_tilde_new[3*i : 3*i + 3] = Yi

        # ---------------------------------------------------
        # 5) desmonta novamente
        # ---------------------------------------------------

        u_tilde_new, v_tilde_new, h_tilde_new = self.desmontar_U(U_tilde_new)

        # ---------------------------------------------------
        # 6) volta ao espaço físico
        # ---------------------------------------------------

        u_new = self.IFFT(u_tilde_new)
        v_new = self.IFFT(v_tilde_new)
        h_new = self.IFFT(h_tilde_new)

        # ---------------------------------------------------
        # 7) remonta vetor físico
        # ---------------------------------------------------

        U_new = self.montar_U(u_new, v_new, h_new)

        return U_new


    def prod_lin_spec(self,x,M):
        '''
        This functions gives the product between operators in the spectral space
        and quantities in the physical space

        Parameters:
        ----------------------------------------------
        M : Operator in the spectral space
        x : quantity in the physical space
        '''

        x_tilde,k_tilde = self.FFT(np.copy(x))
        x_tilde_new = M*x_tilde
        x_new = self.IFFT(x_tilde_new)
        return x_new
    
    def N(self,x,t):
        u,v,h = self.desmontar_U(x)
        n = np.zeros_like(u, dtype = complex)

        u_tilde,k_tilde = self.FFT(u)
        M_line = len(u)
        N_line = int((M_line-1)/2)

        u_redef,k_redef = self.redef_f(f = np.copy(u), k = k_tilde,N = N_line) # redefinir no espaço espectral
        u_deriv = 2*np.pi*k_redef*1j*u_redef  # definimos a derivada no espaço espectral

        w_real = h*self.IFFT(u_deriv)

        N = self.montar_U(u,u,w_real)

        return N

    def FFT(self,f):
        '''
        This function gives the Fourier transformation of a given function. 
        And returns the spectra and the wavenumbers associated 

        Parameters:
        -------------------------------------
        f : function which will be transformated
        '''
        M = len(f)
        f_hat = (2.*np.pi/M)*np.fft.fft(f)

        N = int((M-1)/2)
        k = np.zeros(2 * N + 1, dtype = int)
        k[1:N+1] = np.linspace(1, N, N, endpoint = True)
        k[N+1:] = -np.flip(k[1:N+1])
        return f_hat,k

    def IFFT(self,f):
        '''
        This function gives the inverse Fourier transformation of a given function. 

        Parameters:
        -------------------------------------
        f : function which will be transformated
        '''
        M = len(f)
        f_hat = (M/(2.*np.pi))*np.fft.ifft(f)
        return f_hat

    def redef_f(self,f,k,N):
        '''
        This function aims to induces the anti-aliasing process
        returning zero in the larger wavenumbers.
        
        [x_-2,x_-1,x_0,x_1,x_2] --> [0,x_-1,x_0,x_1,0]

        Parameters:
        ------------------------------------------
        f : spectral components
        k : wavenumbers associated with the spectral components
        N : redefining number
        '''

        N_red = int((N)/2) #aqui definimos a nova discretização para o espaço espactral
        k_red = np.zeros_like(k)
        f_red = np.zeros_like(f)

        k_red[0] = k[0]
        k_red[1:N_red+1] = k[1:N_red+1]
        k_red[-N_red:] = k[-N_red:]

        f_red[0] = f[0]
        f_red[1:N_red+1] = f[1:N_red+1]
        f_red[-N_red:] = f[-N_red:]

        return f_red,k_red

    def trunc_f(self,f,k,N):
        '''
        This function aims to induces the anti-aliasing process
        truncating the extremal wavenumers.
        
        [x_-2,x_-1,x_0,x_1,x_2] --> [x_-1,x_0,x_1]

        Parameters:
        -------------------------------------------------------
        f : spectral components
        k : wavenumbers associated with the spectral components
        N : redefining number
        '''
        N_red = int((N)/2) #aqui definimos a nova discretização para o espaço espactral
        k_trunc = np.zeros(2*N_red + 1,dtype = int)
        f_trunc = np.zeros(2*N_red + 1,dtype = complex)


        k_trunc[0] = k[0]
        k_trunc[1:N_red+1] = k[1:N_red+1]
        k_trunc[-N_red:] = k[-N_red:]

        f_trunc[0] = f[0]
        f_trunc[1:N_red+1] = f[1:N_red+1]
        f_trunc[-N_red:] = f[-N_red:]

        return f_trunc,k_trunc