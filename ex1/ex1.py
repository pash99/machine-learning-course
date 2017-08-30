#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Machine Learning Online Class - Exercise 1: Linear Regression
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import pinv

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def load_data(filename):
    data = list()
    with open(filename, newline='') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            data.append(row)
    data = np.array(data)
    return data[:,:-1], data[:,-1]

def plotData(X, y, theta=None):
    plt.plot(X, y, 'd', markersize=8, markerfacecolor="lightblue",
            label="Training data")
    plt.xlabel("Population of city, 10,000s")
    plt.ylabel("Profit, $10,000s")
    plt.title("Training data")
    if not theta is None:
        plt.plot(X, np.dot(add_intercept_column(X), theta).flatten(),
                 label="Linear regression")
        plt.legend()
        plt.title("Training data and computed linear regression\n"
                  "with parameters $\\theta \\approx [{0[0]:.1f},{0[1]:.1f}]$"
                  .format(theta.flatten()))

def add_intercept_column(X):
    """
    Adds vertical vector of ones as the first column to matrix X to account
    for intercept term theta_0.
    """
    return np.hstack([np.ones((X.shape[0],1)), X])
            
def computeCost(X, y, theta):
    """X should already have an intercept column"""
    m = y.shape[0]
    J = np.dot(X, theta) - y
    J = np.dot(J.T, J)
    J = J / (2*m)
    return J[0][0]

def gradientDescent(X, y, theta, alpha, iterations):
    """y should be vertical vector"""
    m = len(y)
    J_history = np.empty(iterations+1)
    descent_path = np.empty((iterations+1,theta.shape[0]))
    for i in range(iterations):
        descent_path[i] = theta.flatten()
        J_history[i] = computeCost(X, y, theta)
        theta = theta - alpha / m * np.dot(np.dot(X, theta).T - y.T, X).T
    descent_path[-1] = theta.flatten()
    J_history[-1] = computeCost(X, y, theta)
    return theta, J_history, descent_path

def featureNormalize(X):
    """X doesn't have intercept column"""
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
    return np.dot(add_intercept_column(((x-mu)/sigma).reshape(1, -1)),
                  theta)[0,0]

def normalEqn(X, y):
    return np.dot(np.dot(pinv(np.dot(X.T, X)), X.T), y)

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

    for theta in [np.zeros((X1a.shape[1],1)), np.array([[-1], [2]])]:
        # compute cost of theta
        J = computeCost(X1a, y1a, theta)
        print("  for theta = {} the cost J(theta) = {}"
              .format(list(theta.flatten()), J))

    print("Running gradient descent...")

    theta = np.zeros((X1a.shape[1],1))
    alpha = 0.01
    iterations = 1500
    # run gradient descent and compute theta
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
        #prediction = np.dot([[1, population]], theta)[0][0]
        prediction = predict([population], theta)
        print("  For population of {:n} we predict a profit of ${}".format(
            population*10000,
            prediction*10000))
    
    plt.figure()
    plotData(X1, y1, theta)

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
    plt.gca().plot(descent_path[:,0], descent_path[:,1], J_history, "r--",
                   label="Descent path")
    plt.gca().plot([descent_path[ 0,0]], [descent_path[ 0,1]], [J_history[ 0]],
                      color='red', marker='o', label="Starting point")
    plt.gca().plot([descent_path[-1,0]], [descent_path[-1,1]], [J_history[-1]],
                      color='red', marker='x', label="End point")
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    plt.gca().set_zlabel(r"$J(\theta_0,\theta_1)$")
    plt.title("Cost function for univariable linear regression\n"
              "and computed gradient descent path")

    plt.figure()
    plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
    plt.plot(descent_path[0,0], descent_path[0,1], "ro",
             label="Starting point [{:.1f},{:.1f}]".format(
                 descent_path[0,0], descent_path[0,1]))
    plt.plot(descent_path[:,0], descent_path[:,1], "r--", label="Descent path")
    plt.plot(descent_path[-1,0], descent_path[-1,1], "rx",
             label="End point [{:.1f},{:.1f}]".format(
                 descent_path[-1,0], descent_path[-1,1]))
    plt.title("Gradient descent for univariable linear regression\n"
              "with $\\alpha = {}$ and {} iterations".format(alpha, iterations))
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
        theta = np.zeros((X2a.shape[1],1))
        theta, J_history, descent_path = \
                gradientDescent(X2a, y2a, theta, alpha, iterations)
        thetas[alpha] = theta
        print("  For alpha = {} after {} iteration found theta = {} with cost "
              "= {}".format(alpha, iterations, list(theta.flatten()), J_history[-1]))
        plt.plot(range(len(J_history)), J_history, label=alpha)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost $J(\\theta)$")
    plt.title("Rate of gradient descent for various values of $\\alpha$\n"
              "for linear regression")
    plt.legend(title="Value of $\\alpha$:")

    for alpha in [0.01, 1]:
        theta = thetas[alpha]
        print("Theta computed from gradient descent (with alpha = {} and {} "
              "iterations): {}".format(alpha, iterations, list(theta.flatten())))
        print("Predicted price of a 1650 sq-ft, 3 br house (using gradient "
              "descent with alpha = {} and {} iterations): {}".format(alpha,
                  iterations, predict([1650, 3], theta, mu, sigma)))

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
