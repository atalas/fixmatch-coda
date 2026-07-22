# Iterative Teacher-Student implementation

import multiprocessing
import pandas as pd
import numpy as np
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

import augmentations

label_encoder = LabelEncoder()

class ModelData:
	features: np.ndarray
	labels:   np.ndarray
	X:        np.ndarray # reference for function congruency
	y:        np.ndarray # reference for function congruency
						 # augmentation functions work on X and y
						 # so have them point to the necessary
						 # data structure.
	X_labeled:    np.ndarray # input matrix (features)
	y_labeled:    np.ndarray # input labels
	X_unlabeled:  np.ndarray # 50% of input set as unlabeled
	y_unlabeled:  np.ndarray # pseudo labels
	#X_unlab_aug: np.ndarray # augmented unlabeled matrix
	#y_unlab_aug: np.ndarray # augmented pseudo labels
	X_test:   np.ndarray # reserved data to test model
	y_test:   np.ndarray # labels for model testing.
	y_pred:   np.ndarray	
	feature_names:   np.ndarray
	acc_per_loop:    np.ndarray
	max_confidence:  np.ndarray
	tau_per_loop:	 np.ndarray
	min_confidence:  np.ndarray
	percent_confident: np.ndarray
	noise: float
	tau: float
	name: str

def load_data(path, md):
	data = pd.read_csv(path)
	md.features = data.iloc[0:, data.columns != "Group"].to_numpy()
	md.feature_names = data.columns[data.columns != "Group"].tolist()
	md.labels = data['Group'].to_numpy()

def preprocess(md):
    # handle zeros
	pseudo_count = 1.0 
	features_safe = md.features + pseudo_count

	# Calculate the geometric mean across features for each sample
	# (row-wise, axis=1)
	geom_mean = np.exp(np.mean(np.log(features_safe), axis=1, keepdims=True))
	
	# Apply the Centered Log-Ratio (CLR)
	md.features = np.log(features_safe / geom_mean)

	# Encode labels if categorical (e.g., 'A', 'B' -> 0, 1) 
	# if labels.dtype == 'str': 
	md.labels = label_encoder.fit_transform(md.labels)

	# initialize these arrays so that they can be appended to
	md.acc_per_loop = np.array([]) 
	md.tau_per_loop = np.array([]) 
	md.max_confidence = np.array([]) 
	md.min_confidence = np.array([]) 
	md.percent_confident = np.array([]) 

	# Split into train/test sets 
	# First split: 80% data, 20% test (unlabeled)
	X_train_full, md.X_test, y_train_full, md.y_test = train_test_split(
			md.features, md.labels,
			test_size=0.2,
		   	random_state=42, stratify=md.labels
	)

	# We can delete this data, it is no longer needed
	del md.features, md.labels

	# Second split: 50% train, 50% unlabeled
	md.X_labeled, md.X_unlabeled, md.y_labeled, _ = train_test_split(
			X_train_full, y_train_full, test_size=0.5,
		   	random_state=42, stratify=y_train_full)	

#@profile
def TrainingLoop(md):
	#md.tau = .80
	#md.noise = .001
	md.totalLoops = 10
	md.name = "randomforest"

	print(f"Total Loops: {md.totalLoops}")

	# Conservative value for RF because dataset only has 60 samples.
	rf = RandomForestClassifier(
		n_estimators=50,       # Start conservative
		max_depth=7,           # Shallow trees
		min_samples_split=5,   # Require more samples to split
		min_samples_leaf=2,    # Avoid tiny leaves
		random_state=42,
		class_weight="balanced" # If classes are imbalanced
	)

	# initialize pseudo labels.
	# This is only needed if augmentation requires it
	# pseudo_labels = np.zeros(len(md.X_unlabeled), dtype=int)
	# md.y_unlabeled = pseudo_labels
	# md.y = md.y_unlabeled			# X & y references keep augmentation
	md.X = md.X_unlabeled			# functions generic

	# The Teacher - train on the labeled data
	rf.fit(md.X_labeled, md.y_labeled)

	# We can test the augmentation before the loop
	# md = augmentations.compositionalCutmix(md)
	# md.y_pred = rf.predict(md.X_test)
	# return rf

	# initialize the combined arrays
	X_combined = md.X_labeled
	y_combined = md.y_labeled 

	for loop in range(md.totalLoops):
		# Weak Augmentation (for pseudo-labeling)
		# We only want to slightly perturb the data	
		augmentations.augmentTabular(md, noise_std=md.noise)

		# DEBUG: Output the augmented data to a tsv file 
		# np.savetxt('output.tsv', md.X_augmented, delimiter='\t', fmt='%.8e')

		# Predict on unlabeled data
		pseudo_unlab = rf.predict_proba(md.X_augmented)
		# outputArray(pseudo_unlab)
		md.y = np.argmax(pseudo_unlab, axis=1)
		# outputArray(md.y)
		confidences = np.max(pseudo_unlab, axis=1)
		md.max_confidence = np.append(md.max_confidence, np.max(confidences))
		md.min_confidence = np.append(md.min_confidence, np.min(confidences))
		md.percent_confident = np.append(
			md.percent_confident, np.mean(confidences >= md.tau)) # * 100 

		# outputArray(confidences)

		# Filter high-confidence pseudo-labels
		mask = confidences >= md.tau
		# Use strongly augmented data for training
		X_pseudo = md.X[mask]
		y_pseudo = md.y[mask]

		#if(loop == (md.totalLoops - 1)):
			#print (mask)

		# Combine labeled + pseudo-labeled data
		X_combined = np.vstack([X_combined, X_pseudo])
		y_combined = np.concatenate([y_combined, y_pseudo])

		# Train on combined data
		rf.fit(X_combined, y_combined)

		# Decay tau over time 
		# md.tau_per_loop = np.append(md.tau_per_loop, md.tau)
		# md.tau = max(0.7, md.tau - 0.002)
		# md.tau = md.tau - .002

	
		md.y_pred = rf.predict(md.X_test)
		# Keep track of accuracies for plotting
		acc = accuracy_score(md.y_test, md.y_pred);
		md.acc_per_loop = np.append(md.acc_per_loop, acc)
		print(f"Loop: {loop} \t Accuracy: {acc:.2f}")

	# md.y_pred = rf.predict(md.X_test)
	return rf


def createPlot(md):
	font = {'family': 'serif', 'size': 8}

	plt.clf()
	plt.plot(md.min_confidence, "b-", linewidth=1, label="Min Confidence")
	plt.plot(md.max_confidence, "r-", linewidth=1, label="Max Confidence")
	plt.plot(md.acc_per_loop,   "g-", linewidth=1, label="Accuracy")
	# plt.plot(md.tau_per_loop,	"y-", linewidth=1, label="Tau")
	plt.plot(md.percent_confident, "y-", linewidth=1,
		label="Percent Confident")
	
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy/Confidence %")
	plt.title("Per iteration Tau =" + str(md.tau) + " - Noise = " + str(md.noise))
	plt.legend(loc='best')
	plt.grid(True, alpha=0.3)
	# from 0 to total loop in steps of ....
	plt.xticks(np.arange(0, md.totalLoops + 1,
		(md.totalLoops if md.totalLoops < 10 else md.totalLoops / 10)))
	# Save the plot to an image file
	now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
	plt.savefig("rf-T" + str(md.tau) + "-N" + str(md.noise) + ".png", dpi=300, bbox_inches='tight')


def displayMetrics(md):
	print(md.name)

	# Classification metrics
	print(f"Accuracy: {accuracy_score(md.y_test, md.y_pred):.2f}")

	createConfusionMatrix(md)
	
	# Regression metrics 
	# print(f"RMSE: {mean_squared_error(md.y_test, md.y_pred, squared=False):.2f}")
	

def createConfusionMatrix(md):
	cm = confusion_matrix(md.y_test, md.y_pred)
	plt.figure(figsize=(10, 6))

	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
			xticklabels=label_encoder.classes_,
			yticklabels=label_encoder.classes_)
	
	plt.title("Confusion Matrix")
	plt.xlabel("Predicted")
	plt.ylabel("True")
	starttime = datetime.now().strftime("%H%M%S")
	plt.savefig(md.name + ".cm." + starttime + ".png")


def rfFeatureImportance(md, classifier):
	featImp = pd.DataFrame({
		"Feature": md.feature_names,
		"Importance": classifier.feature_importances_,
	}).sort_values("Importance", ascending=False)

	print("\n=== Top Random Forest Important Features ===")
	print(featImp.head(10))

	# Plot feature importance
	plt.figure(figsize=(10, 6))
	sns.barplot(x="Importance", y="Feature", data=featImp.head(10))
	plt.title("Top 10 Important Features")

	# Save the plot to an image file
	starttime = datetime.now().strftime("%Y-%m-%d-%H%M%S")
	plt.savefig(md.name + ".featimp." + starttime + ".png", dpi=300, bbox_inches='tight')

def outputArray(arr):
	now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
	np.savetxt(now + ".tsv", arr, delimiter='\t', fmt='%.8e')

def printTime(msg):
	starttime = datetime.now().strftime("%Y-%m-%d-%H%M%S")
	print(msg + ": " + starttime)

def main():
	infile = ""

	if len(sys.argv) > 1:
		infile = sys.argv[1]
	else:
		print("File name ommited");
		return

	printTime("Process Start")
	md = ModelData()
	load_data(infile, md)
	preprocess(md)

	#printTime("Random Forest start")
    
	md.tau = .80
	md.noise = .001
	rf = TrainingLoop(md)
	createPlot(md)
	md.tau = .85
	md.noise = .001
	rf = TrainingLoop(md)
	createPlot(md)
	md.tau = .90
	md.noise = .001
	rf = TrainingLoop(md)
	createPlot(md)
	md.tau = .95
	md.noise = .001
	rf = TrainingLoop(md)
	createPlot(md)

	md.tau = .80
	md.noise = .01
	rf = TrainingLoop(md)
	createPlot(md)
	md.tau = .85
	md.noise = .01
	rf = TrainingLoop(md)
	createPlot(md)
	md.tau = .90
	md.noise = .01
	rf = TrainingLoop(md)
	createPlot(md)
	md.tau = .95
	md.noise = .01
	rf = TrainingLoop(md)
	createPlot(md)

	md.tau = .80
	md.noise = .1
	rf = TrainingLoop(md)
	createPlot(md)
	md.tau = .85
	md.noise = .1
	rf = TrainingLoop(md)
	createPlot(md)
	md.tau = .90
	md.noise = .1
	rf = TrainingLoop(md)
	createPlot(md)
	md.tau = .95
	md.noise = .1
	rf = TrainingLoop(md)
	createPlot(md)

	# rfFeatureImportance(md, rf)
	# displayMetrics(md)
	#printTime("Random Forest End")

	printTime("Process End")


main()


