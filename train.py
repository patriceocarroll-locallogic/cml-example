from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import mlflow
from mlflow.models.signature import infer_signature

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")

with mlflow.start_run() as run:
    # Fit a model
    depth = 10
    clf = RandomForestClassifier(max_depth=depth)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    signature = infer_signature(X_test, y_pred)

    acc = clf.score(X_test, y_test)
    print(acc)
    with open("metrics.txt", "w") as outfile:
        outfile.write("Accuracy: " + str(acc) + "\n")

    # Plot it
    disp = ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, normalize="true", cmap=plt.cm.Blues
    )
    plt.savefig("confusion_matrix.png")
    mlflow.log_param("max_depth", depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_artifact(local_path="confusion_matrix.png", artifact_path="figures")
    mlflow.sklearn.log_model(clf, artifact_path="sklearn-model", signature=signature, registered_model_name="sk-learn-random-forest-reg-model",)
