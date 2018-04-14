# Train the data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from preprocessing import *
from sklearn.externals import joblib
from time import time

features, labels = return_features_labels() # Return the features and labels
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=3, shuffle=True)


intial = time()


param_gird = [
		{ "n_estimators": [10,100,1000] }
]


r = RandomForestClassifier()
clf = GridSearchCV(r, param_gird) 
clf.fit(features_train, labels_train)
clf_score = clf.best_score_
clf = clf.best_estimator_
print ("The score is: ", clf_score)

print ("Training Time: ", round((time() - intial),3))

# Predictions
predictions = clf.predict(features_test)

cm = confusion_matrix(labels_test, predictions)
report = classification_report(labels_test, predictions)
print()
print ("Confusion matrix: ")
# Printing the confusion confusion_matrix
print (cm)

print()
print ("Printing classification_report")
print (report)

# Saves other details to be used in the web interface
def save_model_param():
	columns = data.columns # features to be tested on
	result = "is_win" # label to be returned

	f = open("../tic_tac_toe/model_data.txt","w")
	f.seek(0)
	f.write(result+"\n")
	for i in columns:
		f.write(i+"\n")
	f.close()


	print ("Model data written successfully.")


# Saves the model
def save_model():
	choice = input("Do you want to save the model: (y/n)")
	if choice == "y":
		# Saving model
		file_name = "../tic_tac_toe/random_forest_classifier.pkl"
		joblib.dump(clf, file_name)
		save_model_param()
		print ("The model is saved.")
	else:
		print ("Okay! Try again")



save_model()
