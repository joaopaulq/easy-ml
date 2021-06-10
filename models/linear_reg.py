import numpy as np

from models.linear_model import LinearModel


class LinearRegression(LinearModel):
    """Classe para o modelo Regressão Linear.
   
    Exemplo de uso:
        > clf = LinearRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_valid)
    """

    def fit(self, x, y):
        """Ajusta o modelo de acordo com os dados de treinamento.

        Args:
            x: Conjunto de dados treinamento. Dim (m, n).
            y: Rótulos de cada exemplo em x. Dim (m,).
        """
        self.theta = np.zeros(x.shape[1])
 
        if self.solver is None or self.solver == 'lstsq':
            self.least_squares(x, y)
        else:
            raise NotImplementedError(f"Método {self.solver} não implementado.")

    
    def least_squares(self, x, y):
        """Método dos mínimos quadrados.
       
        Args:
            x: Conjunto de dados treinamento. Dim (m, n).
            y: Rótulos de cada exemplo em x. Dim (m,).        
        """
        inv = np.linalg.pinv(x.T @ x)
        self.theta = inv @ x.T @ y
        

    def predict(self, x):
        """Faz previsões para um conjunto de dados x.
        
        Args:
            x: Conjunto de dados. Dim (m, n).
        
        Returns:
            h_x: Previsão para cada exemplo em x. Dim (m,).
        """
        return x @ self.theta
    

    def loss(self, y, h_x):
        """Função que mede a performace do modelo.

        Args:
            y: Valores alvo. Dim (m,).
            h_x: Valores previsto. Dim (m,).
        
        Returns:
            J: O quão perto h_x está de y. Escalar.
        """
        # Erro quadrático
        return 0.5 * np.sum(np.square(h_x - y))