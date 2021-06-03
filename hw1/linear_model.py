class LinearModel(object):
    """Classe base para os modelos lineares."""

    def __init__(self, lr=0.2, max_iter=100, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            lr: Taxa de aprendizagem.
            max_iter: Número máximo de iterações.
            eps: Limiar para determinar convergência.
            theta_0: Estimativa inicial para theta. Se "None", usa o vetor zero.
            verbose: Printa o valor da perda durante o treinamento.
        """
        self.theta = theta_0
        self.lr = lr
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose


    def fit(self, x, y):
        """Ajusta o modelo de acordo com os dados de treinamento fornecidos.

        Args:
            x: Conjunto de dados treinamento. Dimensão (m, n).
            y: Rótulos de cada exemplo em x. Dimensão (m,).
        """
        raise NotImplementedError(
            'Subclasses de LinearModel devem implementar o método fit.')


    def predict(self, x):
        """Faz previsões para um conjunto de dados x.
        
        Args:
            x: Conjunto de dados. Dimensão (m, n).
        
        Returns:
            Previsão para cada exemplo em x. Dimensão (m,).
        """
        raise NotImplementedError(
            'Subclasses de LinearModel devem implementar o método predict.')
    

    def loss(y, y_hat):
        """Uma função que mede a performace do modelo.

        Args:
            y: Valores alvo.
            y_hat: Valores previsto.
        
        Returns:
            O quão perto y_hat está de y.
        """
        raise NotImplementedError(
            'Subclasses de LinearModel devem implementar o método loss.')