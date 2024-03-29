import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from algorithm import Perceptron, AdalineGD, AdalineSGD


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)



def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    # print(xx1.shape)
    # print(np.array([xx1.ravel(), xx2.ravel()]).T.shape)
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # print(z.shape)
    z = z.reshape(xx1.shape)
    # print(z.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)




def main():
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    # plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    # plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    # plt.xlabel('petal length')
    # plt.ylabel('sepal length')
    # plt.show()

    # using Perceptron classifier
    # ppn = Perceptron(eta=0.1, n_iter=10)
    # ppn.fit(X, y)
    # plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    # plt.xlabel('Epochs')
    # plt.ylabel('Number of misclassificatoins')
    # plt.show()

    # plot_decision_regions(X, y, classifier=ppn)
    # plt.xlabel('sepal length [cm]')
    # plt.ylabel('petal length [cm]')
    # plt.legend(loc='upper left')
    # plt.show()

    # using AdalineGD classifier
    X_std = np.copy(X)
    X_std[:, 0] = (X_std[:, 0] - X[:, 0].mean()) / X_std[:, 0].std()
    X_std[:, 1] = (X_std[:, 1] - X[:, 1].mean()) / X_std[:, 1].std()

    # ada = AdalineGD(n_iter=15, eta=0.01)
    # ada.fit(X_std, y)
    # plot_decision_regions(X_std, y, classifier=ada)
    # plt.title('Adaline - Gradient Descent')
    # plt.xlabel('sepal length [standarlized]')
    # plt.ylabel('petal length [standarlized]')
    # plt.legend(loc='upper left')
    # plt.show()
    # plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    # plt.xlabel('Epochs')
    # plt.ylabel('Sum-squared-error')
    # plt.show()

    # using AdalineSGD classifier
    ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('sepal length [standarlized]')
    plt.ylabel('petal length [standarlized]')
    plt.legend(loc='upper left')
    plt.show()
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.show()


if __name__ == "__main__":
    main()