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
        """Ajusta o modelo de acordo com os dados de treinamento fornecidos.

        Args:
            x: Conjunto de dados treinamento. Dimensão (m, n).
            y: Rótulos de cada exemplo em x. Dimensão (m,).
        """
        # *** START CODE HERE ***
        _, n = x.shape

        if self.theta is None:
            self.theta = np.zeros(n)

        for i in range(self.max_iter):
            h_x = self.predict(x)
            J = self.loss(y, h_x)
            dJ = np.dot(h_x - y, x.T)
            theta_prev = self.theta
            self.theta = self.theta - self.lr*dJ
            
            if np.allclose(self.theta, theta_prev, atol=self.eps):
                break

            if self.verbose and i % 10:
                print(f"Perda na iteração {i}: {J}")
        # *** END CODE HERE ***


    def predict(self, x):
        """Faz previsões para um conjunto de dados x.
        
        Args:
            x: Conjunto de dados. Dimensão (m, n).
        
        Returns:
            Previsão para cada exemplo em x. Dimensão (m,).
        """
        # *** START CODE HERE ***
        return np.dot(self.theta.T, x)
        # *** END CODE HERE ***
    

    def loss(y, y_hat):
        """Uma função que mede a performace do modelo.

        Args:
            y: Valores alvo.
            y_hat: Valores previsto.
        
        Returns:
            O quão perto y_hat está de y.
        """
        # *** START CODE HERE ***
        # Least squares error.
        return 0.5 * np.linalg.norm(y_hat - y)
        # *** END CODE HERE ***