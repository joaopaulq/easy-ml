class LinearModel(object):
    """Classe base para os modelos lineares."""

    def __init__(self, solver=None, lr=0.2, max_iter=100,
                 eps=1e-5, theta_0=None, verbose=True):
        """
        Args:
            solver: Método de ajuste.
            lr: Taxa de aprendizagem.
            max_iter: Número máximo de iterações.
            eps: Limiar para determinar convergência.
            theta_0: Estimativa inicial para theta. Se "None", usa o vetor zero.
            verbose: Printa o valor da perda durante o treinamento.
        """
        self.solver = solver
        self.lr = lr
        self.max_iter = max_iter
        self.eps = eps
        self.theta = theta_0
        self.verbose = verbose


    def fit(self, x, y):
        """Ajusta o modelo de acordo com os dados de treinamento.

        Args:
            x: Conjunto de dados treinamento. Dim (m, n).
            y: Rótulos de cada exemplo em x. Dim (m,).
        """
        raise NotImplementedError(
            'Subclasses de LinearModel devem implementar o método fit.')


    def predict(self, x):
        """Faz previsões para um conjunto de dados x.
        
        Args:
            x: Conjunto de dados. Dim (m, n).
        
        Returns:
            h_x: Previsão para cada exemplo em x. Dim (m,).
        """
        raise NotImplementedError(
            'Subclasses de LinearModel devem implementar o método predict.')
    

    def loss(self, y, h_x):
        """Função que mede a performace do modelo.

        Args:
            y: Valores alvo. Dim (m,).
            h_x: Valores previsto. Dim (m,).
        
        Returns:
            J: O quão perto h_x está de y. Escalar.
        """
        raise NotImplementedError(
            'Subclasses de LinearModel devem implementar o método loss.')
