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
	features: pd.DataFrame
	labels:   pd.DataFrame
	X:        pd.DataFrame # reference for function congruency
	y:        pd.DataFrame # reference for function congruency
						   # augmentation function workon X and y
						   # thus have them point to the necessary
						   # data structure.
	X_lab:    pd.DataFrame # input matrix (features)
	y_lab:    pd.DataFrame # input labels
	X_unlab:  pd.DataFrame # 50% of input set as unlabeled
	y_unlab:  pd.DataFrame # pseudo labels
	X_unlab_aug: pd.DataFrame # augmented unlabeled matrix
	y_unlab_aug: pd.DataFrame # augmented pseudo labels
	X_test:   pd.DataFrame # reserved data to test model
	y_test:   pd.DataFrame # labels for model testing.
	y_pred:   np.ndarray 
	name:     str

def load_data(path, md):
	data = pd.read_csv(path)
	md.features = data.iloc[0:, data.columns != "Group"]
	md.labels = data['Group']
	return md

def preprocess(md):
	# Handle missing values (if any)
	md.features = md.features.fillna(0)  # Replace NaN with 0 (or use imputation)

	# Normalize features (e.g., log-transform for counts)
	# md.features = np.log1p(md.features)  # log(x+1) to avoid log(0)

	# Encode labels if categorical (e.g., 'A', 'B' -> 0, 1) 
	# if labels.dtype == 'str': 
	md.labels = label_encoder.fit_transform(md.labels) 

	# Split into train/test sets 
	# First split: 80% data, 20% test (unlabeled)
	X_train_full, md.X_test, y_train_full, md.y_test = train_test_split(
			md.features, md.labels,
			test_size=0.2,
		   	random_state=42, stratify=md.labels
	)

	# Second split: 50% train, 50% unlabeled
	md.X_lab, md.X_unlab, md.y_lab, _ = train_test_split(
			X_train_full, y_train_full, test_size=0.5,
		   	random_state=42, stratify=y_train_full)

	return md

	# Standardize features
	# scaler = StandardScaler()
	# md.X_train = scaler.fit_transform(md.X_train)
	# md.X_test = scaler.transform(md.X_test)

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


def fixmatchLoop(md):
	tau = .9
	totalLoops = 50
	md.name = "randomforest"

	# Conservative value for RF because dataset only has 60 samples.
	rf = RandomForestClassifier(
		n_estimators=50,       # Start conservative
		max_depth=7,           # Shallow trees
		min_samples_split=5,   # Require more samples to split
		min_samples_leaf=2,    # Avoid tiny leaves
		random_state=42,
		class_weight="balanced" # If classes are imbalanced
	)

	# initialize pseudo labels
	pseudo_labels = np.zeros(len(md.X_unlab), dtype=int)
	md.y_unlab = pseudo_labels
	md.y = md.y_unlab			# X & y references keep augmentation
	md.X = md.X_unlab			# functions generic

	# Train on the labeled data
	rf.fit(md.X_lab, md.y_lab)

	# We can test the augmentation before the loop
	# md = augmentations.compositionalCutmix(md)
	# md.y_pred = rf.predict(md.X_test)
	# return

	for loop in range(totalLoops):
		# Augment unlabeled data using pseudo-labels (strong aug)
		md = augmentations.compositionalCutmix(md)
		# md = augmentations.aitchisonMixup(md)

		# Predict on unlabeled data
		pseudo_unlab  = rf.predict_proba(md.X)
		md.y = np.argmax(pseudo_unlab, axis=1)
		confidences   = np.max(pseudo_unlab, axis=1)

		# Filter high-confidence pseudo-labels
		mask = confidences >= tau
		X_pseudo = md.X[mask]
		y_pseudo = md.y[mask]

		# Combine labeled + pseudo-labeled data
		X_combined = np.vstack([md.X_lab, X_pseudo])
		y_combined = np.concatenate([md.y_lab, y_pseudo])

		# Train on combined data
		rf.fit(X_combined, y_combined)

		# Optional: Decay tau over time (e.g., tau = max(0.7, 0.9 - 0.02*loop)
		#tau = max(0.7, 0.9 - 0.02 * loop)

	md.y_pred = rf.predict(md.X_test)
	return md, rf


def rfFeatureImportance(md, classifier):
	featImp = pd.DataFrame({
		"Feature": md.features.columns,
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
	md = load_data(infile, md)
	md = preprocess(md)

	printTime("Random Forest start")
	md, rf = fixmatchLoop(md)
	#rfFeatureImportance(md, rf)
	#displayMetrics(md)
	printTime("Random Forest End")

	printTime("Process End")


main()


