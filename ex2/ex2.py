#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Machine Learning Online Class - Exercise 2: Logistic Regression
"""

import numpy as np
from numpy import array, dot, zeros, ones
import csv
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import minimize
import matplotlib.lines


def load_data(filename):
    data = list()
    with open(filename, newline='') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            data.append(row)
    data = np.array(data)
    return data[:,:-1], data[:,-1].astype(int)


def plotData(X, y, label1=None, label2=None):
    plt.scatter(X[y==1,0], X[y==1,1], marker="+", c="b", label=label1)
    plt.scatter(X[y==0,0], X[y==0,1], marker="o", c="w", label=label2,
            edgecolor="g", s=15)


def plotDecisionBoundary(X, y, theta, color="blue", label="Decision boundary"):
    X1_min = np.min(X[:,0])
    X1_max = np.max(X[:,0])
    X1_margin = (X1_max - X1_min) * 0.1
    X2_min = np.min(X[:,1])
    X2_max = np.max(X[:,1])
    X2_margin = (X2_max - X2_min) * 0.1
    if theta.shape[0] == 3:
        #plot_x = np.array([np.min(X[:,0])-1, np.max(X[:,0])+1])
        plot_x = np.array([X1_min-X1_margin, X1_max+X1_margin])
        plot_y = (-1) / theta[2,0] * (theta[1,0] * plot_x + theta[0,0])
        return plt.plot(plot_x, plot_y, color=color, label=label)
    elif theta.shape[0] > 3:
        degree = get_mapFeature_degree(theta.size)
        #u = np.linspace(-1, 1.5, 51)
        u = np.linspace(X1_min-X1_margin, X1_max+X1_margin, 51)
        v = np.linspace(X2_min-X2_margin, X2_max+X2_margin, 51)
        z = np.empty((u.size, v.size))
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                z[i,j] = dot(mapFeature(ui, vj, degree=degree), theta)
        z = z.T
        return plt.contour(u, v, z, [0], colors=color, alpha=0.5, label=label)


def add_intercept_column(X):
    """
    Adds vertical vector of ones as the first column to matrix X to account
    for intercept term theta_0.
    """
    assert np.ndim(X) > 0
    if np.ndim(X) < 2:
        intercept = ones(1)
    else:
        intercept = ones((np.shape(X)[0],1))
    return np.hstack([intercept, X])


def sigmoid(X):
    return 1 / (1 + np.exp((-1) * X))


def costFunction(theta, X, y, lambda_=None, gradFlatten=False):
    m = y.shape[0]
    theta = theta.reshape((-1,1))
    h = sigmoid(dot(X, theta))
    J = (-1) / m * (dot(y.T, np.log(h)) + dot((1-y.T), np.log(1-h)))
    grad = 1 / m * dot(X.T, (h-y))
    if lambda_ is not None:
        J = J + lambda_ / (2*m) * dot(theta[1:].T, theta[1:])
        grad[1:] += lambda_ / m * theta[1:]
    if gradFlatten:
        # some scalar function minimization solvers require that you return
        # grad flattened
        grad = grad.flatten()
    return J[0,0], grad


def mapFeature(X1, X2, degree):
    """
    Feature mapping function to polynomial features. Returns a new feature
    array with more features, comprising of 1, X1, X2, X1**2, X1*X2, X2**2,
    X1**3, X1**2*X2, X1*X2**2, etc.
 
    Inputs X1, X2 must be the same size.
    """
    assert np.shape(X1) == np.shape(X2)
    X1 = np.array(X1)
    X2 = np.array(X2)
    if np.ndim(X1) < 2:
        X1 = X1.reshape((-1,1))
        X2 = X2.reshape((-1,1))
    X = ones(X1.shape, dtype=X1.dtype)
    for i in range(1,degree+1):
        for j in range(i+1):
            X = np.hstack([X, X1**(i-j) * X2**j])
    return X


def get_mapFeature_degree(number_of_columns):
    """
    Return the degree with which features in matrix of number_of_columns were
    mapped using mapFeature function. That is, if

        Xn = mapFeature(X1, X2, n)

    then
        
        get_mapFeature_degree(Xn.shape[1]) == n

    Note:
        num_of_columns = (degree+1) * (degree+2) / 2
        degree = (sqrt(8*num_of_columns + 1) - 3) / 2
    """
    return int(((8*number_of_columns + 1)**(1/2) - 3) / 2)


def predict(theta, X):
    # x should not containt intercept column
    p = sigmoid(dot(add_intercept_column(X), theta))
    p[p >= 0.5] = 1
    p[p <  0.5] = 0
    return p.astype(int)


def plotModel(X, y, theta, label1=None, label2=None):
    assert np.shape(X)[1] == 2
    X1 = X[:,0]
    X2 = X[:,1]
    X1_cover = [np.min(X1), np.max(X1)]
    X2_cover = [np.min(X2), np.max(X2)]
    X1_vals = np.linspace(X1_cover[0]-2, X1_cover[1]+2, 101)
    X2_vals = np.linspace(X2_cover[0]-2, X2_cover[1]+2, 101)
    h_vals = zeros((len(X1_vals), len(X2_vals)))
    degree = get_mapFeature_degree(np.shape(theta)[0])
    for i, x1 in enumerate(X1_vals):
        for j, x2 in enumerate(X2_vals):
            h_vals[i,j] = \
                sigmoid(dot(mapFeature(x1, x2, degree), theta)[0])
    h_vals = h_vals.T

    x1, x2 = np.meshgrid(X1_vals, X2_vals)
    plt.gca(projection="3d")
    plt.gca().plot_surface(x1, x2, h_vals, cmap=matplotlib.cm.coolwarm)
    plt.gca().scatter(X1[y == 1], X2[y==1], 1, label=label1)
    plt.gca().scatter(X1[y == 0], X2[y==0], 0, label=label2)


if __name__ == "__main__":
    print("Exercise 2: logistic regression.")
    print("================================")
    print()

    print("Loading first dataset...")
    X1, y1 = load_data("ex2data1.txt")

    print("Dataset contains", X1.shape[0], "examples with", X1.shape[1],
          "feature(s).")
        
    print("First 5 examples from the dataset:")
    print(np.hstack([X1[:5], y1[:5].reshape(-1,1)]))
    print()

    #print("Plotting data...")
    plt.figure()
    plotData(X1, y1, label1="Admitted", label2="Not admitted")
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    #plt.legend(["Admitted", "Not admitted"])
    plt.legend()
    plt.title("First dataset: applicants' admission to university")
    #plt.show()

    print("Testing cost function...")
    
    X1a = add_intercept_column(X1)
    y1a = y1.reshape(-1, 1)
    m1, n1 = X1a.shape

    for theta, expected_cost, expected_gradient in zip(
            [zeros((n1,1)), np.array([[-24], [0.2], [0.2]])],
            [0.693,0.218],
            [[-0.1000, -12.0092, -11.2628], [0.043, 2.566, 2.647]]):
        cost, grad = costFunction(theta, X1a, y1a)
        print("For theta = {}:".format(list(theta.flatten())))
        print("  cost = {} (expected approx. {})".format(cost, expected_cost))
        print("  gradient = {} (expected approx. {})"
              .format(list(grad.flatten()), expected_gradient))
        #print("Cost at initial theta (zeros):", cost);
        #print("Expected cost (approx): ");
        #print("Gradient at initial theta (zeros):", list(grad.T[0]));
        #print("Expected gradients (approx): ");

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_tnc.html
    theta_init = zeros((n1,1))
    result = minimize(
            x0=theta_init,
            fun=costFunction,
            args=(X1a,y1a),
            method="TNC",
            jac=True)
    theta_min1 = result.x.reshape(n1,-1)
    cost_min1 = result.fun

    print("Computed gradient descent using the truncated Newton (TNC) algorithm"
          " (fmin_tnc):")
    print("  theta =", list(theta_min1.flatten()), "(expected approx. "
          "[-25.161, 0.206, 0.201]).")
    print("  cost:", cost_min1, "(expected approx. 0.203)")

    print("For a student with scores 45 and 85, we predict an admission "
          "probability of {} (expected value: 0.775 +/- 0.002)"
          .format(sigmoid(dot([1, 45, 85], theta_min1))[0]))

    print("Prediction accuracy on the train set: {:.1%} (expected: 89.0%)"
          .format(np.mean(predict(theta_min1, X1) == y1a)))

    #print("Plotting decision boundary...")
    plt.figure()
    plotData(X1, y1, label1="Admitted", label2="Not admitted")
    plotDecisionBoundary(X1, y1, theta_min1)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    legend_handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=legend_handles[1:]+legend_handles[0:1])
    plt.title("Training data and computed decision boundary")

    #print("Plotting logistic model...")
    plt.figure()
    plotModel(X1, y1, theta_min1, label1="Exam 1 score", label2="Exam 2 score")
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.gca().set_zlabel("$h_\\theta$")
    plt.title("Logistic model with parameter $\\theta \\approx "
              "[{theta[0]:.1f}, {theta[1]:.1f}, {theta[2]:.1f}]$"
              .format(theta=list(theta_min1.flatten())))

    #plt.show()

    print()
    print("Loading second dataset...")
    X2, y2 = load_data("ex2data2.txt")

    print("Dataset contains", X2.shape[0], "examples with", X2.shape[1],
          "feature(s).")
        
    print("First 5 examples from the dataset:")
    print(np.hstack([X2[:5], y2[:5].reshape(-1,1)]))
    print()

    #print("Plotting data...")
    plt.figure()
    plotData(X2, y2, label1="Passed QA", label2="Failed QA")
    plt.xlabel("Microchip test 1")
    plt.ylabel("Microchip test 2")
    plt.legend()
    plt.title("Second dataset: microchip quality assurance")
    #plt.show()

    X2a = mapFeature(X2[:,0].reshape(-1,1), X2[:,1].reshape(-1,1), 6)
    y2a = y2.reshape(-1,1)
    m2, n2 = X2a.shape

    print("Testing cost function with regularization parameter:")
    for theta_init, lambda_, exp_cost, exp_grad in zip(
            [zeros((n2,1)), ones((n2,1))],
            [1, 10],
            [0.693, 3.16],
            [[0.0085, 0.0188, 0.0001, 0.0503, 0.0115],
             [0.3460, 0.1614, 0.1948, 0.2269, 0.0922]]):
        cost, grad = costFunction(theta_init, X2a, y2a, lambda_)
        print("For theta = {} and lambda = {}:"
              .format(list(theta.flatten()), lambda_))
        print("  J(theta, lambda) = {} (expected approx. {})"
              .format(cost, exp_cost))
        print("  first 5 values of gradient at theta: {} (expected approx. {})"
              .format(list(grad[:5].flatten()), exp_grad))

    print("Testing prediction accuracy on train set for various values of "
          "regularization parameter lambda:")
    theta_init = zeros((n2,1))
    lambdas = [0, 1, 10, 100]
    colors = ["orange", "blue", "black", "red"]
    thetas = {}
    plt.figure()
    plotData(X2, y2, label1="Passed QA", label2="Failed QA")
    #legend_handles, _ = plt.gca().get_legend_handles_labels()
    legend_handles = []
    for lambda_, color in zip(lambdas, colors):
        cost, grad = costFunction(theta_init, X2a, y2a, lambda_)
        result = minimize(
                x0=theta_init,
                fun=costFunction,
                args=(X2a,y2a,lambda_),
                method="TNC",
                jac=True)
        theta = result.x.reshape(n2,-1)
        cost = result.fun
        precision = np.mean(predict(theta, X2a[:,1:]) == y2a)
        thetas[lambda_] = {"theta_min": theta,
                           "cost_min": cost,
                           "precision": precision}
        print("  lambda = {}: accuracy = {:%}{}".format(
            lambda_,
            precision,
            "" if lambda_ != 1 else " (expected approx. 83.1%)"
            ))
        plotDecisionBoundary(X2, y2, theta, color=color)
        legend_handles.append(matplotlib.lines.Line2D([], [], color=color,
            alpha=0.5,
            label="$\\lambda = {}$ (accur. {:.0%})".format(lambda_, precision)))
    theta_min2 = thetas[1]["theta_min"]
    cost_min2 = thetas[1]["cost_min"]
    plt.xlabel("Microchip test 1")
    plt.ylabel("Microchip test 2")
    plt.legend(handles=legend_handles)
    plt.title("Decision boundaries and prediction accuracies\n"
              "for various values of regularization parameter $\\lambda$")

    print("Plotting data...")
    plt.show()
