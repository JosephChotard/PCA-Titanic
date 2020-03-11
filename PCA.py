import matplotlib.pyplot as plt

def doPCA(data):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(data)
    return pca


data = [[0,2,3,4],
        [4,5,6,7]]
pca = doPCA(data)
print(pca.explained_variance_ratio_)
pc0 = pca.components_[0]
print(pc0)
pc1 = pca.components_[1]
print(pc1)
plt.xlabel("x")
plt.ylabel("y")
plt.show()