from pre_train.reader import get_data
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

npdata, _ = get_data(['aalborg.csv'])

print(npdata.shape)
# print(npdata[0])
# print(data[0])
import matplotlib.pyplot as plt
for i in range(2, 3):
    pca = PCA(n_components=i, svd_solver='full')
    res = pca.fit_transform(npdata[:, 3:])
    # print(res)
    # print(pca.explained_variance_)
    # print(pca.explained_variance_ratio_)
    # print(pca.singular_values_)
    tot = sum(pca.explained_variance_ratio_)
    print('for i={0} we have taken var={1}'.format(i, tot))

    x = res[:, 0]
    y = res[:, 1]

    # print(pca.get_covariance())

    emb = TSNE(n_components=2, verbose=1, n_iter=1000, perplexity=30).fit_transform(npdata[3:])
    # print(emb)
    #
    # x = emb[:, 0]
    # y = emb[:, 1]

    plt.scatter(x, y)
    plt.show()


