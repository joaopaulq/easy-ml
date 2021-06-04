import numpy as np 

from linear_model import LinearModel 


class Perceptron(LinearModel):
    """Classe para o modelo Perceptron.
   
    Exemplo de uso:
        > clf = Perceptron()
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
            dJ = x.T @ (y - h_x)
            theta_prev = self.theta
            self.theta = self.theta + self.lr*dJ
            
            if np.allclose(self.theta, theta_prev, atol=self.eps):
                break

            if self.verbose and i % 10:
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
        h_x = x @ self.theta
        return np.where(h_x >= 0, 1, 0)
        # *** END CODE HERE ***
    

    def loss(y, h_x):
        """Uma função que mede a performace do modelo.
    
        Args:
            y: Valores alvo. Dim (m,).
            h_x: Valores previsto. Dim (m,).
        
        Returns:
            J: O quão perto h_x está de y. Escalar.
        """
        # *** START CODE HERE ***
        # 0-1 loss function.
        return np.sum(y == h_x)
        # *** END CODE HERE ***