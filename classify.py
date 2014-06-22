# Constants
NUMBER_OF_COMPONENTS = 200 # the remaining number of components after dimension reduction

import sys

if(len(sys.argv) != 3):
	print("Usage: python classify.py <train_database> <test_database>")
	exit()

TRAIN_PATH = sys.argv[1]
TEST_PATH = sys.argv[2]

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
