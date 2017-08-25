#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import dot

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def load_data(filename):
    with open(filename) as f:
        data = np.array(pd.read_csv(f, header=None))
    return data[:,:-1], data[:,-1]

def plotData(X, y):
    plt.plot(X, y, 'd', markersize=8, markerfacecolor="lightblue", label="Training data")
    plt.xlabel("Population of city, 10,000s")
    plt.ylabel("Profit, $10,000s")
    plt.title("Training data")

def computeCost(X, y, theta):
    """X should have first columnt made of 1s"""
    m = y.shape[0]
    J = np.dot(X, theta) - y
    J = np.dot(J.T, J)
    J = J / (2*m)
    return J[0][0]

def gradientDescent(X, y, theta, alpha, iterations):
    """y should be vertical vector, that is array of shape (m,1)"""
    m = len(y)
    J_history = np.empty(iterations)
    descent_path = np.empty((iterations,theta.shape[0]))
    for i in range(iterations):
        descent_path[i] = theta.T[0]
        theta = theta - alpha / m * np.dot(np.dot(X, theta).T - y.T, X).T
        J_history[i] = computeCost(X, y, theta)
    return theta, J_history, descent_path

def featureNormalize(X):
    """X doesn't have first row of 1"""
    m = X.shape[0]
    n = X.shape[1]
    mu = np.zeros(n)
    sigma = np.ones(n)
    #X_norm = np.empty((m,n))
    for i in range(n):
        mu[i] = X[:,i].mean()
        sigma[i] = X[:,i].std()
        #X_norm[:,i] = (X[:,i] - mu[i].T) / sigma[i].T
    X_norm = (X - np.tile(mu, (m,1))) / np.tile(sigma, (m,1))
    return X_norm, mu, sigma

def predict(x, theta, mu=None, sigma=None):
    if mu is None:
        mu = np.zeros(len(x))
    if sigma is None:
        sigma = np.ones(len(x))
    return np.dot(np.hstack([np.ones((1,1)), ((x-mu)/sigma).reshape(1, -1)]), theta)[0,0]

def normalEqn(X, y):
    from numpy import dot
    from numpy.linalg import pinv
    return dot(dot(pinv(dot(X.T, X)), X.T), y)

if __name__ == "__main__":
    print("Loading first dataset...")

    X1, y1 = load_data("ex1data1.txt")
    m1 = len(y1)
    n1 = X1.shape[1]

    print("Dataset contains", m1, "examples with", n1, "feature(s).")

    print("Plotting data...")

    plotData(X1, y1)
    #plt.show()
    
    print("Testing cost function...")

    # add vertical vector as the first column of X1
    X1a = np.hstack([np.ones((X1.shape[0],1)), X1])
    y1a = y1.reshape(-1,1)

    theta = np.zeros((n1+1,1))
    J = computeCost(X1a, y1a, theta)
    print("With theta = {} cost J(theta) = {}".format(theta.T[0], J))

    theta = np.array([[-1], [2]])
    J = computeCost(X1a, y1a, theta)
    print("With theta = {} cost J(theta) = {}".format(theta.T[0], J))

    print("Running gradient descent...")

    alpha = 0.01
    iterations = 1500
    theta_start = np.zeros((n1+1,1))
    theta, J_history, descent_path = gradientDescent(X1a, y1a, theta_start, alpha, iterations)

    print("For alpha = {alpha} after {iterations} iterations of gradient "
          "descent found theta = {theta}, J(theta) = {J}".format(
              theta=theta.T[0],
              J=computeCost(X1a, y1a, theta),
              alpha=alpha,
              iterations=iterations))
    
    plt.figure()
    plotData(X1, y1)
    plt.plot(X1a[:,1], np.dot(X1a, theta).T[0], label="Linear regression $\\theta = [{:.2f},{:.2f}]$".format(theta.T[0,0], theta.T[0,1]))
    plt.legend()
    plt.title("Training data and computed linear regression")

    for population in [3.5, 7]:
        prediction = np.dot([[1, population]], theta)[0][0]
        print("For population of {:n} we predict a profit of {}".format(
            population*10000,
            prediction*10000))

    print('Visualizing J(theta_0, theta_1)...')

    theta0_vals = np.linspace(-10, 10, 100);
    theta1_vals = np.linspace(-1, 4, 100);

    J_vals = np.zeros((len(theta0_vals),len(theta1_vals)))
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            J_vals[i,j] = computeCost(
                    X1a,
                    y1a,
                    np.array([[theta0_vals[i]], [theta1_vals[j]]]))
    J_vals = J_vals.T

    plt.figure()
    th0, th1 = np.meshgrid(theta0_vals, theta1_vals)
    plt.gca(projection="3d")
    plt.gca().plot_surface(th0, th1, J_vals, cmap=cm.coolwarm)
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")

    plt.figure()
    plt.plot(theta_start[0,0], theta_start[1,0], "ro", label="Starting point")
    plt.plot(descent_path[:,0], descent_path[:,1], "r--", label="Descent path")
    plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    plt.plot(theta[0,0], theta[1,0], "rx", markersize=10, linewidth=2, label="Final point")
    plt.title("Gradient descent with $\\alpha = {}$ "
              "for {} iterations".format(alpha, iterations))
    plt.legend()

    #plt.show()
    print()

    print("Loading second dataset...")

    X2, y2 = load_data("ex1data2.txt")
    m2 = len(y2)
    n2 = X2.shape[1]

    print("Dataset contains", m2, "examples with", n2, "features.")

    print("First 10 examples from the dataset:")
    print(np.hstack([X2[:10], y2[:10].reshape(-1,1)]))
    print()

    print("Normalizing features...")
    X2n, mu, sigma = featureNormalize(X2)

    print("First 10 normalized features (mu = {}, sigma = {}):".format(list(mu), list(sigma)))
    print(X2n[:10])
    print()

    print("Solving with gradient descent...")
    # add vertical vector of ones as the first column of X1
    X2a = np.hstack([np.ones((X2n.shape[0],1)), X2n])
    y2a = y2.reshape(-1,1)

    iterations = 400
    alphas = [1.284, 1, 0.3, 0.1, 0.03, 0.01, 0.003]
    thetas = {}
    plt.figure()
    for alpha in alphas:
        theta = np.zeros((X2a.shape[1],1))
        theta, J_history, _ = \
                gradientDescent(X2a, y2a, theta, alpha, iterations)
        thetas[alpha] = theta
        print(" For alpha = {} after {} iteration found theta = {} with cost = "
              "{}".format(alpha, iterations, list(theta.T[0]), J_history[-1]))
        plt.plot(range(len(J_history)), J_history, label=alpha)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost of linear regression $J(\\theta)$")
    plt.title("Rate of gradient descent for various values of $\\alpha$")
    plt.legend(title="Value of $\\alpha$:")

    for alpha in [0.01, 1]:
        theta = thetas[alpha]
        print("Theta computed from gradient descent (with alpha = {} and {} "
              "iterations): {}".format(alpha, iterations, list(theta.T[0])))
        print("Predicted price of a 1650 sq-ft, 3 br house (using gradient "
              "descent with alpha = {} and {} iterations): {}".format(alpha,
                  iterations, predict(np.array([1650, 3]), theta, mu, sigma)))

    print()
    print("Solving with normal equations...")

    X2a = np.hstack([np.ones((X2.shape[0],1)), X2])
    theta = normalEqn(X2a, y2a)

    print("Theta computed from the normal equations:", list(theta.T[0]))

    print("Predicted price of a 1650 sq-ft, 3 br house (using normal "
          "equations):", predict(np.array([1650, 3]), theta))

    plt.show()
