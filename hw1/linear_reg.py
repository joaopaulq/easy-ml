import numpy as np

from linear_model import LinearModel


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
        # *** START CODE HERE ***
        if self.theta is None:
            _, n = x.shape
            self.theta = np.zeros(n)
            
        for i in range(self.max_iter):
            h_x = self.predict(x)
            J = self.loss(y, h_x)
            dJ = x.T @ (h_x - y)
            theta_prev = self.theta
            self.theta = self.theta - self.lr*dJ
            
            if np.allclose(self.theta, theta_prev, atol=self.eps):
                break

            if self.verbose and i % 10 == 0:
                print(f"Perda na iteração {i}: {J}")
        # *** END CODE HERE ***


    def predict(self, x):
        """Faz previsões para um conjunto de dados x.
        
        Args:
            x: Conjunto de dados. Dim (m, n).
        
        Returns:
            h_x: Previsão para cada exemplo em x. Dim (m,).
        """
        # *** START CODE HERE ***
        return x @ self.theta
        # *** END CODE HERE ***
    

    def loss(self, y, h_x):
        """Uma função que mede a performace do modelo.

        Args:
            y: Valores alvo. Dim (m,).
            h_x: Valores previsto. Dim (m,).
        
        Returns:
            J: O quão perto h_x está de y. Escalar.
        """
        # *** START CODE HERE ***
        # Least squares error.
        return 0.5 * np.linalg.norm(h_x - y)
        # *** END CODE HERE ***