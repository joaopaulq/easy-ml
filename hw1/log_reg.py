import numpy as np 

from util import sigmoid 
from linear_model import LinearModel 


class LogisticRegression(LinearModel):
    """Classe para o modelo Regressão Logística.
    
    Exemplo de uso:
        > clf = LogisticRegression()
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
            dJ = np.dot(y - h_x, x.T)
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
            x: Conjunto de dados. Dimensão (m, n).
        
        Returns:
            y: Previsão para cada exemplo em x. Dimensão (m,).
        """
        # *** START CODE HERE ***
        z = np.dot(self.theta.T, x)
        y = sigmoid(z)
        return np.where(y >= 0.5, 1, 0)
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
        # Cross-entropy loss function.
        return np.sum(y*np.log(y_hat) + (1 - y_hat)*np.log(1 - y_hat)) 
        # *** END CODE HERE ***
