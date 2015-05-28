import cPickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pdb

'''
timing 12.5K

real    88m31.232s
user    59m18.815s
sys     29m5.556s

timing 20k

real    175m1.952s
user    121m4.781s
sys     53m42.382s
'''

X = cPickle.load(open("representation.pkl"))

n_train_samples = 12500 # 25k seg fault

X_pca = PCA(n_components=50).fit_transform(X)
X_train = X_pca[:n_train_samples]

X_train_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(X_train)
cPickle.dump(X_train_embedded, open("tsne-representation.pkl", "w"))
