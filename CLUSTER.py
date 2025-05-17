import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from time import time


class HierarchicalAgglomerativeClustering:
    def __init__(self, n_clusters=5):
        """
        Кластеризатор, использующий только sklearn реализацию

        Параметры:
        n_clusters -- количество кластеров
        """
        self.n_clusters = n_clusters

    def fit_predict(self, X):
        """Основной метод, использующий только sklearn реализацию"""
        model = AgglomerativeClustering(n_clusters=self.n_clusters,
                                        linkage='ward')
        return model.fit_predict(X)


def evaluate_clustering(true_labels, pred_labels, X):
    """Оценка качества кластеризации"""
    print(f"Adjusted Rand Score: {adjusted_rand_score(true_labels, pred_labels):.3f}")
    print(f"Silhouette Score: {silhouette_score(X, pred_labels):.3f}")


    # Тестирование на данных разного размера
sample_sizes = [300, 1500]  # Разные размеры данных

for n_samples in sample_sizes:
    print(f"\n{'=' * 50}\nТестирование на {n_samples} samples\n{'=' * 50}")
    # Генерация данных
    X, y = make_blobs(n_samples=n_samples, n_features=2, centers=5,
                          cluster_std=1.5, random_state=42)

    # Создание модели
    model = HierarchicalAgglomerativeClustering(n_clusters=5)

    # Кластеризация и оценка
    start_time = time()
    pred_labels = model.fit_predict(X)
    elapsed = time() - start_time

    print(f"\nВремя выполнения: {elapsed:.3f} сек")
    evaluate_clustering(y, pred_labels, X)

    # Визуализация
    plt.figure(figsize=(8, 4))
    plt.scatter(X[:, 0], X[:, 1], c=pred_labels, cmap='rainbow', s=20)
    plt.title(f'Результат кластеризации (n_samples={n_samples})')
    plt.show()

    # Дендрограмма (только для небольших данных)
    if n_samples <= 1000:
        plt.figure(figsize=(10, 5))
        linkage_matrix = linkage(X, method='ward')
        dendrogram(linkage_matrix, truncate_mode='lastp', p=15)
        plt.title("Дендрограмма (метод Уорда)")
        plt.show()