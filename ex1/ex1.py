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
import matplotlib.animation
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


def plotData(X, y, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(X, y, 'd', markersize=8, markerfacecolor="lightblue",
            label="Training data")
    ax.set_xlabel("Population of city, 10,000s")
    ax.set_ylabel("Profit, $10,000s")
    ax.set_title("Linear regression: training data")


def plotModel(X, theta, ax=None):
    if ax is None:
        ax = plt.gca()
    X_cover = [[np.min(X)], [np.max(X)]]
    ax.plot(X_cover, dot(add_intercept_column(X_cover), theta),
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
    return np.hstack([ones((np.shape(X)[0], 1)), X])


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


def plotSurface(theta0_vals, theta1_vals, J_vals, descent_path, J_history,
        theta_min=None, cost_min=None):
    th0, th1 = np.meshgrid(theta0_vals, theta1_vals)
    plt.gca(projection="3d")
    plt.gca().plot_surface(th0, th1, J_vals, cmap=matplotlib.cm.coolwarm)
    plt.gca().plot(descent_path[:,0], descent_path[:,1], J_history, "r--",
                   label="Descent path")
    plt.gca().plot([descent_path[0,0]],  [descent_path[0,1]],  [J_history[0]],
                   color='red', marker='o', label="Starting point")
    plt.gca().plot([descent_path[-1,0]], [descent_path[-1,1]], [J_history[-1]],
                   color='red', marker='x', label="End point")
    if theta_min is not None and cost_min is not None:
        plt.gca().plot(theta_min[0], theta_min[1], [cost_min],
                       color='yellow', marker='+', label="Global minimum")
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    plt.gca().set_zlabel(r"$J(\theta_0,\theta_1)$")
    plt.title("Mean squared error cost function\n"
              "and computed gradient descent path")


def plotContour(theta0_vals, theta1_vals, J_vals, theta_min=None, J_min=None,
        ax=None):
    if ax is None:
        ax = plt.gca()
    ax.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")
    if theta_min is not None:
        ax.plot(theta_min[0,0], theta_min[1,0], "b+",
                label="Global minimum"+(" ($J(\\theta) \\approx {:.1f}$)"
                      .format(J_min) if J_min else ""))


def plotContourPath(descent_path, J_history):
    plt.plot(descent_path[0,0], descent_path[0,1], "ro",
             label="Starting point ($J(\\theta) \\approx {:.1f}$)"
                   .format(J_history[0]))
    plt.plot(descent_path[:,0], descent_path[:,1], "r--", label="Descent path")
    plt.plot(descent_path[-1,0], descent_path[-1,1], "rx",
             label="End point ($J(\\theta) \\approx {:.1f}$)"
                   .format(J_history[-1]))
    legend_handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=legend_handles[1:]+legend_handles[0:1])
    plt.title("Gradient descent on cost function")


def plotAnimation(X, y, duration=20, delay=200, theta_init=None,
        alpha=0.005, iterations=1500):
    Xa = add_intercept_column(X)
    ya = np.reshape(y, (-1,1))
    if theta_init is None:
        theta_init = zeros((Xa.shape[1],1))
    num_of_frames = duration * 1000 // delay

    # Run gradient descent and compute theta
    theta, J_history, descent_path = \
        gradientDescent(Xa, ya, theta_init, alpha, iterations)
    theta_min = normalEqn(Xa, ya)
    J_min = computeCost(Xa, ya, theta_min)

    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = zeros((len(theta0_vals), len(theta1_vals)))
    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            J_vals[i,j] = computeCost(X1a, y1a, array([[theta0], [theta1]]))
    J_vals = J_vals.T

    X_cover = [[np.min(X)], [np.max(X)]]

    fig = plt.gcf()
    ax1, ax2, = fig.get_axes()
    fig.suptitle("Model fitting with gradient descent")
    lines1, = ax1.plot([], [], color="orange", animated=True,
                       label="Linear model")
    lines2, = ax2.plot([], [], "r--", animated=True, label="Descent path")
    lines3, = ax2.plot([], [], "rx", animated=True,
                       label="End point ($J(\\theta) \\approx {:.1f}$)"
                             .format(J_history[-1]))
    infobox1 = ax1.text(0.60, 0.02, "", transform=ax1.transAxes)
    infobox2 = ax2.text(0.58, 0.96, "", transform=ax2.transAxes,
                        verticalalignment="top",
                        bbox={"facecolor": "white",
                              "alpha": 0.85,
                              "edgecolor": "lightgray"})

    def ani_init():
        plotData(X, y, ax1)
        ax1.set_title("Linear model")
        plotContour(theta0_vals, theta1_vals, J_vals, theta_min, J_min, ax2)
        ax2.plot(descent_path[0,0], descent_path[0,1], "ro",
                 label="Starting point ($J(\\theta) \\approx {:.1f}$)"
                       .format(J_history[0]))
        ax2.set_title("Gradient descent (learning rate $\\alpha = {}$)"
                      .format(alpha))
        lines1.set_data(X_cover, predict(X_cover, theta))
        return lines1, lines2, lines3, infobox1, infobox2,

    def ani_mate(iteration):
        theta = descent_path[iteration].reshape(-1,1)
        lines1.set_ydata(predict(X_cover, theta))
        lines2.set_data(descent_path[:iteration+1,0],
                        descent_path[:iteration+1,1])
        infobox1.set_text("$\\theta \\approx "
                          "[{theta[0]:+.2f},{theta[1]:+.2f}]$"
                          .format(theta=theta.flatten()))
        infobox2.set_text("Iteration: {iteration:>4}\n"
                          "$\\theta \\approx "
                          "[{theta[0]:+.2f},{theta[1]:+.2f}]$\n"
                          "$J(\\theta) \\approx {cost:.2f}$"
                          .format(iteration=iteration, theta=theta.flatten(),
                                  cost=J_history[iteration]))
        if iteration == len(descent_path)-1:
            lines3.set_data(descent_path[-1,0], descent_path[-1,1])
        return lines1, lines2, lines3, infobox1, infobox2,

    return matplotlib.animation.FuncAnimation(
        fig,
        func=ani_mate,
        init_func=ani_init,
        frames=(np.round( # speed of descent is not linear
            np.linspace(0, (len(descent_path)-1)**(1/3), num_of_frames+1)**3)
            .astype(int)),
        interval=delay,
        repeat=False,
        blit=True)


if __name__ == "__main__":

    print("Exercise 1: Linear regression.")
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

    theta_min = normalEqn(X1a, y1a)
    J_min = computeCost(X1a, y1a, theta_min)

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
        prediction = predict([population], theta)
        print("  For population of {:n} we predict a profit of ${}"
              .format(population*10000, prediction*10000))

    plt.figure()
    plotData(X1, y1)
    plotModel(X1, theta)

    #print('Visualizing J(theta_0, theta_1)...')

    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = zeros((len(theta0_vals), len(theta1_vals)))
    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            J_vals[i,j] = computeCost(X1a, y1a, array([[theta0], [theta1]]))
    J_vals = J_vals.T

    plt.figure()
    plotSurface(theta0_vals, theta1_vals, J_vals, descent_path, J_history)

    plt.figure()
    plotContour(theta0_vals, theta1_vals, J_vals, theta_min, J_min)
    plotContourPath(descent_path, J_history)
    plt.title("Gradient descent on cost function\n"
              "for {} interations with learning rate $\\alpha = {}$"
              .format(iterations, alpha))

    # Plotting Animation
    plt.subplots(1, 2, figsize=(10,4.5))
    ani = plotAnimation(X1, y1, alpha=alpha)
    #plt.gcf().set_tight_layout(True)

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

    # Now solve with normal equation and compare result with the one computed
    # with gradient descent
    print()
    print("Solving with normal equation...")

    X2a = add_intercept_column(X2)
    theta = normalEqn(X2a, y2a)

    print("Theta computed from the normal equation:", list(theta.flatten()))

    print("Predicted price of a 1650 sq-ft, 3 br house (using normal "
          "equation):", predict([1650, 3], theta))

    print("Plotting data...")
    plt.show()

    #print("Saving animation to file...")
    #ani.save("Animation_1.mp4", writer="ffmpeg")
    #ani.save("Animation_1.gif", writer="imagemagick")
    #ani.save("Animation_1.mng", writer="imagemagick")
