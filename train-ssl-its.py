# Iterative Teacher-Student implementation

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
	X_lab:    np.ndarray # input matrix (features)
	y_lab:    np.ndarray # input labels
	X_unlab:  np.ndarray # 50% of input set as unlabeled
	y_unlab:  np.ndarray # pseudo labels
	#X_unlab_aug: np.ndarray # augmented unlabeled matrix
	#y_unlab_aug: np.ndarray # augmented pseudo labels
	X_test:   np.ndarray # reserved data to test model
	y_test:   np.ndarray # labels for model testing.
	y_pred:   np.ndarray	
	feature_names:   np.ndarray
	acc_per_loop:    np.ndarray
	min_confidence:  np.ndarray
	max_confidence:  np.ndarray
	name:     str

def load_data(path, md):
	data = pd.read_csv(path)
	md.features = data.iloc[0:, data.columns != "Group"].to_numpy()
	md.feature_names = data.columns[data.columns != "Group"].tolist()
	md.labels = data['Group'].to_numpy()

def preprocess(md):
	# Handle missing values (if any)
	# Replace NaN with 0 (or use imputation)
	# md.features = md.features.fillna(0)  

	# Normalize features (e.g., log-transform for counts)
	# md.features = np.log1p(md.features)  # log(x+1) to avoid log(0)

	# Encode labels if categorical (e.g., 'A', 'B' -> 0, 1) 
	# if labels.dtype == 'str': 
	md.labels = label_encoder.fit_transform(md.labels)

	# initialize these arrays so that they can be appended to
	md.acc_per_loop = np.array([]) 
	md.max_confidence = np.array([]) 
	md.min_confidence = np.array([]) 

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
	md.X_lab, md.X_unlab, md.y_lab, _ = train_test_split(
			X_train_full, y_train_full, test_size=0.5,
		   	random_state=42, stratify=y_train_full)	

#@profile
def TrainingLoop(md):
	tau = .9
	totalLoops = 20
	md.name = "randomforest"

	print(f"Initial Tau: {tau}")
	print(f"Total Loops: {totalLoops}")

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
	# pseudo_labels = np.zeros(len(md.X_unlab), dtype=int)
	# md.y_unlab = pseudo_labels
	# md.y = md.y_unlab			# X & y references keep augmentation
	md.X = md.X_unlab			# functions generic

	# The Teacher - train on the labeled data
	rf.fit(md.X_lab, md.y_lab)

	# We can test the augmentation before the loop
	# md = augmentations.compositionalCutmix(md)
	# md.y_pred = rf.predict(md.X_test)
	# return rf

	for loop in range(totalLoops):
		# Weak Augmentation (for pseudo-labeling)
		# We only want to slightly perturb the data	
		augmentations.augmentTabular(md)

		# DEBUG: Output the augmented data to a tsv file 
		# np.savetxt('output.tsv', md.X_aug, delimiter='\t', fmt='%.8e')

		# Predict on unlabeled data
		pseudo_unlab = rf.predict_proba(md.X_aug)
		# outputArray(pseudo_unlab)
		md.y = np.argmax(pseudo_unlab, axis=1)
		# outputArray(md.y)
		confidences = np.max(pseudo_unlab, axis=1)
		md.max_confidence = np.append(md.max_confidence, np.max(confidences))
		md.min_confidence = np.append(md.min_confidence, np.min(confidences))
		# outputArray(confidences)

		# Strong augmentation step
		#augmentations.compositionalCutmix(md)

		# Filter high-confidence pseudo-labels
		mask = confidences >= tau
		# Use strongly augmented data for training
		X_pseudo = md.X[mask]
		y_pseudo = md.y[mask]

		#if(loop == (totalLoops - 1)):
			#print (mask)

		# Combine labeled + pseudo-labeled data
		X_combined = np.vstack([md.X_lab, X_pseudo])
		y_combined = np.concatenate([md.y_lab, y_pseudo])

		# Train on combined data
		rf.fit(X_combined, y_combined)

		# Optional: Decay tau over time (e.g., tau = max(0.7, 0.9 - 0.02*loop)
		#tau = max(0.7, 0.9 - 0.02 * loop)
		tau = tau - .05
	
		md.y_pred = rf.predict(md.X_test)
		# Keep track of accuracies for plotting
		acc = accuracy_score(md.y_test, md.y_pred);
		md.acc_per_loop = np.append(md.acc_per_loop, acc)
		print(f"Accuracy: {acc:.2f}")

	# md.y_pred = rf.predict(md.X_test)
	return rf


def createPlot(md):
	plt.clf()
	plt.plot(md.min_confidence, "b-", linewidth=2)
	plt.plot(md.max_confidence, "r-", linewidth=2)
	plt.plot(md.acc_per_loop, "g-", linewidth=2)
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy/Confidence %")
	plt.title("Values per iteration")
	plt.grid(True, alpha=0.3)
	plt.xticks(np.arange(0, 21, 1))
	# Save the plot to an image file
	now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
	plt.savefig(md.name + ".confidence." + now + ".png", dpi=300, bbox_inches='tight')


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
	rf = TrainingLoop(md)
	createPlot(md)
	# rfFeatureImportance(md, rf)
	# displayMetrics(md)
	#printTime("Random Forest End")

	printTime("Process End")


main()


