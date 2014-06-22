# Constants
NUMBER_OF_COMPONENTS = 200 # the remaining number of components after dimension reduction

import sys

if(len(sys.argv) != 4):
	print("Usage: python classify.py <train_database> <test_database> <output_file>")
	exit()

TRAIN_PATH = sys.argv[1]
TEST_PATH = sys.argv[2]
OUTPUT_PATH = sys.argv[3]

from load_data import load_data
print("Loading data...")
x, y = load_data(TRAIN_PATH)
x_test, y_test = load_data(TEST_PATH)

print("Scaling...")
from transforming import scale
x, x_test = scale(x, x_test)

print("Dimension reduction...")
from transforming import dimension_reduction
x, x_test = dimension_reduction(x, x_test, NUMBER_OF_COMPONENTS)

print("Building classifier...")
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="log", penalty="elasticnet", shuffle=True, n_iter=500, class_weight="auto")
clf.fit(x, y)

print("Predicting...")
f = open(OUTPUT_PATH, "w")
for sample in x_test:
	f.write(str(clf.predict_proba(sample)[0][1]))
	f.write("\n")

f.close()

