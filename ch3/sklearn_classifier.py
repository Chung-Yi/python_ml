from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup markers generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 =  np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
    
    # hightlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
    ppn.fit(X_train_std, y_train)

    y_pred = ppn.predict(X_test_std)
    print("Misclassified smaples: %d" % (y_test != y_pred).sum())
    print("Accuracy: %0.2f" % accuracy_score(y_test, y_pred))

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined_std = np.hstack((y_train, y_test))
    print(X_combined_std.shape)
    print(y_combined_std.shape)
    plot_decision_regions(X=X_combined_std, y=y_combined_std, classifier=ppn, test_idx=range(105, 150))
    plt.xlabel('sepal length [standarlized]')
    plt.ylabel('petal length [standarlized]')
    plt.legend(loc='upper left')
    plt.show()


    # Logistic regression
    lr = LogisticRegression(C=1000.0, random_state=0)
    lr.fit(X_train_std, y_train)
    plot_decision_regions(X=X_combined_std, y=y_combined_std, classifier=lr, test_idx=range(105, 150))
    plt.xlabel('sepal length [standarlized]')
    plt.ylabel('petal length [standarlized]')
    plt.legend(loc='upper left')
    plt.show()

    # regularization
    weights, params = [], []
    for c in np.arange(-1, 5):
        lr = LogisticRegression(C=10**float(c), random_state=0)
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(10**float(c))
    
    weights = np.array(weights)
    plt.plot(params, weights[:, 0], label='petal lenghth')
    plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
    plt.xlabel('C')
    plt.ylabel('weight coefficient')
    plt.legend(loc='upper left')
    plt.xscale('log')
    plt.show()


if __name__ == "__main__":
    main()