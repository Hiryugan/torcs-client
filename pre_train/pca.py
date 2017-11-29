from pre_train.reader import get_data2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


npdata, _ = get_data2('forza_2')

print(npdata.shape)
# print(npdata[0])
# print(data[0])
import matplotlib.pyplot as plt
for i in range(2, 3):
    pca = PCA(n_components=i, svd_solver='full',tol=0.5)
    res = pca.fit_transform(npdata[:, 18:37])

    tot = sum(pca.explained_variance_ratio_)
    print('for i={0} we have taken var={1}'.format(i, tot))

    x = res[:, 0]
    y = res[:, 1]

    # print(pca.get_covariance())

    emb = TSNE(n_components=2, verbose=1, n_iter=1000, perplexity=30).fit_transform(npdata[:, 18:37])
    # print(emb)
    #
    x = emb[:, 0]
    y = emb[:, 1]

    plt.scatter(x, y)
    plt.show()


