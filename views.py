from django.http import HttpResponse
from django.shortcuts import render_to_response
import numpy as np
from sklearn.externals import joblib


# Entry point
def home(request):
	return render_to_response("index.html")

# Retrains the classifier (Simple Http responses)
def retrain_classifier(request):
	return HttpResponse("Run train.py or (Training in browser comming soon...)")

# Returns the current accuracy of the classifier (Simple HttpResponses)
def current_score(request):
	model = load_model()
	return HttpResponse("Score is: 0.9701")

def load_model():
	try:
		model = joblib.load("../tic_tac_toe/random_forest_classifier.pkl")
		print ("Model successfully loaded.")
		return model
	except:
		print ("Error")

def request_and_predict(request):

	# Gathering values
	x1 = request.GET.get("first_row_left","")
	x2 = request.GET.get("first_row_middle","")
	x3 = request.GET.get("first_row_right",""),

	y1 = request.GET.get("center_row_left","")
	y2 = request.GET.get("center_row_middle","")
	y3 = request.GET.get("center_row_right","")


	z1 = request.GET.get("bottom_row_left","")
	z2 = request.GET.get("bottom_row_middle","")
	z3 = request.GET.get("bottom_row_right","")

	points = [x1, x2, x3, y1,y2, y3, z1,z2,z3]
	stz = [] # Main List to be used
	# Converting points into 1 and 0

	for i in points:
		if isinstance(i, tuple): # As we have a tuple present in the list, we check the instance to get the first element (anomaly)
			if i[0] == 'x':
				stz.append(1)
			else:
				stz.append(0)
		else:
			if i == 'x':
				stz.append(1)
			else:
				stz.append(0)


	data_points = np.array(stz)


	print(stz)
	model = load_model() # Loading model
	stz = np.array(stz) # Converting into np array
	# Predicion
	output = model.predict(stz.reshape(1,-1)) # Reshaping data to get ndim = 2
	if output == 1: # 1 stands for positive
		if np.max(stz) == 1: # Checking the maximum value as the winner will always contain the max value
			return HttpResponse("WIN (x)")
		else:
			return HttpResponse("WIN (o)")
	else:
		return HttpResponse("LOSE")



