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
        """Ajusta o modelo de acordo com os dados de treinamento.

        Args:
            x: Conjunto de dados treinamento. Dim (m, n).
            y: Rótulos de cada exemplo em x. Dim (m,).
        """
        _, n = x.shape

        if self.theta is None:
            self.theta = np.zeros(n)
        else:
            assert self.theta.shape == n

        if self.solver is None or self.solver == "newton":
            self.newtons_method(x, y) 
        elif self.solver == "gradient":
            self.gradient_descent(x, y)
        else:
            raise NotImplementedError(f"Método {self.solver} não implementado.")


    def gradient_ascent(self, x, y):
        """Método de gradiente ascendente.
       
        Args:
            x: Conjunto de dados treinamento. Dim (m, n).
            y: Rótulos de cada exemplo em x. Dim (m,).        
        """
        assert self.max_iter >= 0 and self.lr > 0

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
    

    def newtons_method(self, x, y):
        """Método de Newton-Raphson.
       
        Args:
            x: Conjunto de dados treinamento. Dim (m, n).
            y: Rótulos de cada exemplo em x. Dim (m,).        
        """
        pass 


    def predict(self, x):
        """Faz previsões para um conjunto de dados x.
        
        Args:
            x: Conjunto de dados. Dim (m, n).
        
        Returns:
            h_x: Previsão para cada exemplo em x. Dim (m,).
        """
        z = x @ self.theta
        return sigmoid(z)
    

    def loss(self, y, h_x):
        """Uma função que mede a performace do modelo.

        Args:
            y: Valores alvo. Dim (m,).
            h_x: Valores previsto. Dim (m,).
        
        Returns:
            J: O quão perto h_x está de y. Escalar.
        """
        # Cross-entropy loss function.
        return np.sum(y*np.log(h_x) + (1-y)*np.log(1-h_x))
