import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math

def predict_using_sklearn():
    d=pd.read_csv("test_scores.csv")
    r=LinearRegression()
    r.fit(d[['math']],d.cs)
    return r.coef_, r.intercept_

def gradient_descent(x,y):
    m_curr=0
    b_curr=0
    iterations=1000
    n=len(x)
    learning_rate=0.01
    cost_previous=0

    for i in range(iterations):
        y_predicted=m_curr * x + b_curr
        cost = (1/n)*sum((y - y_predicted)**2)
        md=-(2/n)*sum(x*(y-y_predicted))
        bd=-(2/n)*sum(y-y_predicted)
        m_curr=m_curr - learning_rate * md
        b_curr=b_curr - learning_rate * bd
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous=cost
        if i%100==0:
          print ("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost, i))
    return m_curr, b_curr

if __name__ == "__main__":
    d=pd.read_csv("test_scores.csv")
    x=np.array(d.math)
    y=np.array(d.cs)

    m, b = gradient_descent(x,y)
    print("Using gradient descent function: Coef {} Intercept {}".format(m, b))

    m_sklearn, b_sklearn = predict_using_sklearn()
    print("Using sklearn: Coef {} Intercept {}".format(m_sklearn,b_sklearn))