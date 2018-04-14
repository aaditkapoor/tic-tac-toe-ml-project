import matplotlib.pyplot as plt
from preprocessing import *


features, labels = return_features_labels()


# Showing graphs
for i in data.columns:
	plt.scatter(data[i].values, labels)
	plt.show()
