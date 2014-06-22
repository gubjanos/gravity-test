# Constants
NUMBER_OF_COMPONENTS = 200 # the remaining number of components after dimension reduction

from load_data import load_data
print("Loading data...")
x, y = load_data("c:\\gravity\\valami_train_no_duplicates.txt")
x_test, y_test = load_data("c:\\gravity\\valami_test.txt")

print("Scaling...")
from transforming import scale
x, x_test = scale(x, x_test)

print("Dimension reduction...")
from transforming import dimension_reduction
x, x_test = dimension_reduction(x, x_test, NUMBER_OF_COMPONENTS)
