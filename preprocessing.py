# Author: Aadit Kapoor
# preprocess the data.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

# Loading data
data = pd.read_csv("../tic-tac-toe.data.txt", sep = ",")
data_copy = pd.read_csv("../tic-tac-toe.data.txt", sep = ",") # for further use


# View the data
data.head()


# Assigning columns
data.columns = ["first_row_left", "first_row_middle", "first_row_right", "center_row_left", "center_row_middle", "center_row_right", "bottom_row_left", "bottom_row_middle", "bottom_row_right", "is_win"]
data_copy.columns = ["first_row_left", "first_row_middle", "first_row_right", "center_row_left", "center_row_middle", "center_row_right", "bottom_row_left", "bottom_row_middle", "bottom_row_right", "is_win"]



def return_features_labels():
	global data
	global data_copy
	# As we can see the the different move options, we perform label encoding.
	mapping_for_moves = {'x':1, "o":0} # For b, we put mean of the data.
	mapping_for_wins = {"positive":1, "negative":0} # Positive is win, negative is lose
	data.is_win = data.is_win.map(mapping_for_wins)
	data_copy.is_win = data_copy.is_win.map(mapping_for_wins)

	data = data.drop(columns=["is_win"], axis=1)

	for i in data.columns: # Applying map to all the columns except is_win.
	    data[i] = data[i].map(mapping_for_moves)


	# Extracting features and labels
	features = data.values
	labels = data_copy.is_win.values

	# Filling missing values aka "b" with the mean
	features = (Imputer().fit_transform(features))


	features = features.astype(np.int)
	labels = labels.astype(np.int)

	return features, labels




