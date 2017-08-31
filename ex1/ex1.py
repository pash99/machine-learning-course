#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Machine Learning Online Class - Exercise 1: Linear Regression
"""

import csv

import numpy as np
from numpy import array, dot, zeros, ones
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D

def load_data(filename):
    data = list()
    with open(filename, newline='') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            data.append(row)
    data = array(data)
    return data[:,:-1], data[:,-1]

def plotData(X, y, theta=None):
    plt.plot(X, y, 'd', markersize=8, markerfacecolor="lightblue",
             label="Training data")
    plt.xlabel("Population of city, 10,000s")
    plt.ylabel("Profit, $10,000s")
    plt.title("Linear regression: training data")
    if theta is not None:
        plt.plot(X, dot(add_intercept_column(X), theta).flatten(),
                 label="Linear model")
        plt.legend()
        plt.title("Training data and estimated linear regression model\n"
                  "with parameter $\\theta \\approx [{0[0]:.1f},{0[1]:.1f}]$"
                  .format(theta.flatten()))

def add_intercept_column(X):
    """
    Adds vertical vector of ones as the first column to matrix X to account
    for intercept term theta_0.
    """
    return np.hstack([ones((X.shape[0], 1)), X])

def computeCost(X, y, theta):
    """X should already have an intercept column"""
    m = y.shape[0]
    J = dot(X, theta) - y
    J = dot(J.T, J)
    J = J / (2*m)
    return J[0][0]

def gradientDescent(X, y, theta, alpha, iterations):
    """y should be vertical vector"""
    m = len(y)
    J_history = np.empty(iterations+1)
    descent_path = np.empty((iterations+1, theta.shape[0]))
    for i in range(iterations):
        descent_path[i] = theta.flatten()
        J_history[i] = computeCost(X, y, theta)
        theta = theta - alpha / m * dot(dot(X, theta).T - y.T, X).T
    descent_path[-1] = theta.flatten()
    J_history[-1] = computeCost(X, y, theta)
    return theta, J_history, descent_path

def featureNormalize(X):
    """X doesn't have intercept column"""
    m = X.shape[0]
    n = X.shape[1]
    mu = zeros(n)
    sigma = ones(n)
    #X_norm = np.empty((m,n))
    for i in range(n):
        mu[i] = X[:,i].mean()
        sigma[i] = X[:,i].std()
        #X_norm[:,i] = (X[:,i] - mu[i].T) / sigma[i].T
    X_norm = (X - np.tile(mu, (m,1))) / np.tile(sigma, (m,1))
    return X_norm, mu, sigma

def predict(X, theta, mu=None, sigma=None):
    X_ndim = np.ndim(X)
    if X_ndim < 2:
        # If X is a number or simple list, reshape (and convert) it into
        # 2-dimensional numpy.array
        X = array(X, ndmin=2)
    if mu is None:
        mu = zeros(np.shape(X), dtype=int)
    else:
        mu = np.tile(mu, (np.shape(X)[0], 1))
    if sigma is None:
        sigma = ones(np.shape(X), dtype=int)
    else:
        sigma = np.tile(sigma, (np.shape(X)[0], 1))
    # np.array(X) -- explicitly convert X into numpy.array in case it's a
    # Python list
    result = dot(add_intercept_column(((np.array(X)-mu)/sigma)),
                 np.reshape(theta, (-1,1)))
    if X_ndim < 2:
        return result.flatten()[0]
    else:
        return result.flatten()

def normalEqn(X, y):
    return dot(dot(pinv(dot(X.T, X)), X.T), y)

if __name__ == "__main__":

    print("Exercise 1: linear regression.")
    print("==============================")
    print()

    print("Loading first dataset...")

    X1, y1 = load_data("ex1data1.txt")
    m1 = len(y1)
    n1 = X1.shape[1]

    print("Dataset contains", m1, "examples with", n1, "feature(s).")

    print("First 10 examples from the dataset:")
    print(np.hstack([X1[:10], y1[:10].reshape(-1,1)]))
    print()

    #print("Plotting data...")

    plotData(X1, y1)
    #plt.show()

    print("Testing cost function...")

    # add intercept column
    X1a = add_intercept_column(X1)
    # reshape into a vertical vector
    y1a = y1.reshape(-1,1)

    for theta in [zeros((X1a.shape[1], 1)), array([[-1], [2]])]:
        # compute cost of theta
        J = computeCost(X1a, y1a, theta)
        print("  for theta = {} the cost J(theta) = {}"
              .format(list(theta.flatten()), J))

    print("Running gradient descent...")

    theta = zeros((X1a.shape[1],1))
    alpha = 0.01
    iterations = 1500
    # Run gradient descent and compute theta
    theta, J_history, descent_path = \
        gradientDescent(X1a, y1a, theta, alpha, iterations)

    assert list(descent_path[-1]) == list(theta.flatten())
    assert J_history[-1] == computeCost(X1a, y1a, theta)

    print("For alpha = {alpha} after {iterations} iterations of gradient "
          "descent found:\n"
          "  theta = {theta}\n"
          "  J(theta) = {J}"
          .format(
              theta=theta.flatten(),
              J=computeCost(X1a, y1a, theta),
              alpha=alpha,
              iterations=iterations))

    print("Testing prediction...")
    for population in [3.5, 7]:
        #prediction = dot([[1, population]], theta)[0][0]
        prediction = predict([population], theta)
        print("  For population of {:n} we predict a profit of ${}"
              .format(population*10000, prediction*10000))

    plt.figure()
    plotData(X1, y1, theta)

    print('Visualizing J(theta_0, theta_1)...')

    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    J_vals = zeros((len(theta0_vals), len(theta1_vals)))
    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            J_vals[i,j] = computeCost(X1a, y1a, array([[theta0], [theta1]]))
    J_vals = J_vals.T

    plt.figure()
    th0, th1 = np.meshgrid(theta0_vals, theta1_vals)
    plt.gca(projection="3d")
    plt.gca().plot_surface(th0, th1, J_vals, cmap=matplotlib.cm.coolwarm)
    plt.gca().plot(descent_path[:,0], descent_path[:,1], J_history, "r--",
                   label="Descent path")
    plt.gca().plot([descent_path[0,0]],  [descent_path[0,1]],  [J_history[0]],
                   color='red', marker='o', label="Starting point")
    plt.gca().plot([descent_path[-1,0]], [descent_path[-1,1]], [J_history[-1]],
                   color='red', marker='x', label="End point")
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    plt.gca().set_zlabel(r"$J(\theta_0,\theta_1)$")
    plt.title("Mean squared error cost function\n"
              "and computed gradient descent path")

    plt.figure()
    plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
    plt.plot(descent_path[0,0], descent_path[0,1], "ro",
             label="Starting point ($J(\\theta) \\approx {:.1f}$)"
                   .format(J_history[0]))
    plt.plot(descent_path[:,0], descent_path[:,1],  "r--",
             label="Descent path")
    plt.plot(descent_path[-1,0], descent_path[-1,1], "rx",
             label="End point ($J(\\theta) \\approx {:.1f}$)"
                   .format(J_history[-1]))
    plt.title("Gradient descent on cost function")
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
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

    # Normalize feature values
    print("Normalizing features...")
    X2n, mu, sigma = featureNormalize(X2)

    print("First 10 normalized features (mu = {}, sigma = {}):"
          .format(list(mu), list(sigma)))
    print(X2n[:10])
    print()

    print("Solving with gradient descent...")
    # add intercept column
    X2a = add_intercept_column(X2n)
    # reshape into vertical vector
    y2a = y2.reshape(-1,1)

    # compute theta with gradient descent for various values of alpha
    iterations = 400
    alphas = [1.284, 1, 0.3, 0.1, 0.03, 0.01, 0.003]
    thetas = dict()
    plt.figure()
    for alpha in alphas:
        theta = zeros((X2a.shape[1],1))
        theta, J_history, descent_path = \
            gradientDescent(X2a, y2a, theta, alpha, iterations)
        thetas[alpha] = theta
        print("  For alpha = {} after {} iteration found theta = {} with cost "
              "= {}"
              .format(alpha, iterations, list(theta.flatten()), J_history[-1]))
        plt.plot(range(len(J_history)), J_history, label=alpha)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost $J(\\theta)$")
    plt.title("Rate of gradient descent for various\n"
              "values of learning rate parameter $\\alpha$")
    plt.legend(title="Value of $\\alpha$:")

    for alpha in [0.01, 1]:
        theta = thetas[alpha]
        print("Theta computed from gradient descent (with alpha = {} and {} "
              "iterations): {}"
              .format(alpha, iterations, list(theta.flatten())))
        print("Predicted price of a 1650 sq-ft, 3 br house (using gradient "
              "descent with alpha = {} and {} iterations): {}"
              .format(alpha, iterations, predict([1650, 3], theta, mu, sigma)))

    # Now solve with normal equation and compare with the one computed with
    # gradient descent
    print()
    print("Solving with normal equation...")

    X2a = add_intercept_column(X2)
    theta = normalEqn(X2a, y2a)

    print("Theta computed from the normal equation:", list(theta.flatten()))

    print("Predicted price of a 1650 sq-ft, 3 br house (using normal "
          "equation):", predict([1650, 3], theta))

    plt.show()
