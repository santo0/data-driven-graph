import numpy as np
import pandas as pd
import cvxpy as cp
import networkx as nx
from sklearn.covariance import graphical_lasso, empirical_covariance
from graph import draw_graph, Node, get_graph
from data import get_nodes, get_scaled_data
from graph import draw_graph, Node, get_graph
from data import get_nodes


def euclidean_distance(nodes, threshold, std):
    m = len(nodes)
    adj_mx = np.zeros(shape=(m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            dis = Node.distance(nodes[i], nodes[j])
            if i != j and dis < threshold:
                adj_mx[i, j] = np.exp(-(dis**2)/(2*std**2))
    return adj_mx

# df is expected to be scale


def glasso(df, alpha, threshold, max_iter=100):
    numerical_columns = df.columns[1:]
    S = empirical_covariance(df[numerical_columns])
    W, O = graphical_lasso(S, alpha=alpha, max_iter=max_iter)
    # set diagonal to zero
    for i in range(W.shape[0]):
        W[i][i] = 0
    # delete small weights
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i][j] < threshold:
                W[i][j] = 0
    print(W)
    return W


def glasso_opt(df, lamb, threshold):
    numerical_columns = df.columns[1:]
    n = len(numerical_columns)
    S = empirical_covariance(df[numerical_columns])
    Theta = cp.Variable((n, n), 'theta', symmetric=True)
    obj = cp.Minimize(
        cp.trace(S@Theta)-cp.log_det(Theta) +
        lamb*cp.sum(cp.abs(cp.multiply(Theta, np.ones((n, n)) - np.identity(n))))
    )
    const = [Theta >> np.zeros((n, n))]
    prob = cp.Problem(obj, const)
    prob.solve()

    W = Theta.value
    # set diagonal to zero
    for i in range(W.shape[0]):
        W[i][i] = 0
    # delete small weights
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i][j] < threshold:
                W[i][j] = 0
    print(W)
    return W


def laplacian(df, alpha, beta, fixed_L=None):
    min_threshold = 0.009
    numerical_columns = df.columns[1:]
    X = df[numerical_columns].to_numpy().transpose()
    n, m = np.shape(X)
    if not fixed_L:
        L = cp.Variable((n, n), "L")
        Y = X.copy()
    else:
        L = fixed_L
        Y = cp.Variable((n, m), "Y")

    obj = cp.Minimize(
        cp.norm(X-Y, "fro")**2 +
        alpha * cp.trace(cp.quad_form(Y, L)) +
        beta*cp.norm(L, "fro")**2
    )
    const = [
        cp.trace(L) == n,
        L == L.T,
        cp.multiply(L, np.ones((n, n)) - np.identity(n)) <= 0,
        L@np.ones((n, 1)) == 0,
    ]

    # for i in range(n):
    #    for j in range(n):
    #        if i != j:
    #            const.append(L[i, j] <= 0)

    prob = cp.Problem(obj, const)
    prob.solve(solver=cp.ECOS)
    return L
    if fixed_L:
        return Y.value
    else:
        Lvalue = L.value
        # delete small weights
        for i in range(Lvalue.shape[0]):
            for j in range(Lvalue.shape[1]):
                if -Lvalue[i][j] > -min_threshold:
                    Lvalue[i][j] = 0
        return Lvalue
