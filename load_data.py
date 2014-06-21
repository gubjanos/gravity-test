def load_data(path):
    from numpy import loadtxt
    instances = loadtxt(path)
    y = []
    x = []
    for j in xrange(len(instances)):
        y.append(instances[j][0])
        x.append(instances[j][1:])
    return x, y
