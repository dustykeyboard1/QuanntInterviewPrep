import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_ndarray(X, labels=None, title="PCA Projection"):
    """
    Plot a 2D or 3D numpy array of shape (n_samples, n_features).
    - If 2D: scatter plot.
    - If 3D: 3D scatter plot.

    Args:
        X (ndarray): shape (n_samples, 2) or (n_samples, 3).
        labels (ndarray or list, optional): class/cluster labels for coloring.
        title (str): plot title.
    """
    n_samples, n_features = X.shape
    assert n_features in (2, 3), "Can only plot 2D or 3D arrays."

    color_bar = False if labels is None else True
    if labels is None:
        labels = np.zeros(n_samples)  # single color if no labels

    fig = plt.figure(figsize=(6, 6))

    if n_features == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=40)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    elif n_features == 3:
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap="viridis", s=40)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

    ax.set_title(title)
    if color_bar:
        fig.colorbar(scatter, ax=ax, shrink=0.6, label="Label")
    plt.show()


def scale_df(df, col_names):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df[col_names]))


def covariance_matrix(df):
    return np.cov(df.T)


def eigen_decomposition(cov_matrix, k):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_variance = np.sum(eigenvalues)

    eigenvalues = eigenvalues[:k]
    eigenvectors = eigenvectors[:, :k]

    explained_variance = eigenvalues / total_variance
    print(f"Explained Variance: {explained_variance}")
    print(f"Cumulative Explained Variance: {np.sum(explained_variance)}")
    return eigenvalues, eigenvectors


def project_data(df, eigenvectors):
    return np.dot(df, eigenvectors)


if __name__ == "__main__":
    k = 3
    df = pd.read_csv("PCAdata.csv")
    cols = ["Feat. 1", "Feat. 2", "Feat. 3", "Feat. 4", "Feat. 5"]
    scaled_df = scale_df(df, cols)
    cov_matrix = covariance_matrix(scaled_df)
    evals, evecs = eigen_decomposition(cov_matrix, k)
    projected_data = project_data(scaled_df, evecs)
    if k in [2, 3]:
        plot_ndarray(projected_data, title="PCA Projection of Scaled Data")
