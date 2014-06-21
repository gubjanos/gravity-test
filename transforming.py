def scale(train, test):
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return train, test

def dimension_reduction(train, test, n_comp):   
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_comp,whiten=True).fit(train)
    train_pca = pca.transform(train)
    test_pca = pca.transform(test)
    return train_pca, test_pca
