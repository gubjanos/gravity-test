from load_data import load_data
print("Loading data...")
x, y = load_data("../valami_train_no_duplicates.txt")

from sklearn.cross_validation import train_test_split
x, x_test, y, y_test = train_test_split(x,y, test_size=0.25)

print("Scaling...")
from transforming import scale
x, x_test = scale(x, x_test)

print("Dimension reduction...")
from transforming import dimension_reduction
x, x_test = dimension_reduction(x, x_test, 200)

from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.qda import QDA
from sklearn.svm import SVC

classifiers = [SGDClassifier(loss="log", penalty="l1", class_weight="auto"),
		DecisionTreeClassifier(),
		GaussianNB(),
		LDA(),
		KNeighborsClassifier(),
		AdaBoostClassifier(),
		QDA(),
		SVC(kernel="linear", C=0.025),
		SVC(gamma=2, C=1)]

from sklearn.metrics import roc_auc_score

for clf in classifiers:
	clf.fit(x,y)
	predictions = []
	for sample in x_test:
		predictions.append(clf.predict_proba(sample)[0][1])
	print(clf)
	print(roc_auc_score(y_test,predictions))

