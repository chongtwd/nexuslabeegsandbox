import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score

df = pd.read_csv("eeg_spectra.csv")

xs = df.iloc[:, 5:]
ys = np.array([0 if x == "C" else 1 for x in df["Group"]])

xtrain, xtest, ytrain, ytest = train_test_split(xs, ys, test_size=0.2)

lr = LogisticRegression()
lr.fit(xtrain, ytrain)
y_hat = lr.predict(xtest)
y_probs = lr.predict_proba(xtest)[:, 1]
fpr, tpr, thresholds = roc_curve(ytest, y_probs, pos_label=1)
print("Precision : {}".format(precision_score(ytest, y_hat)))
print("Recall : {}".format(recall_score(ytest, y_hat)))
print("accuracy : {}".format(accuracy_score(ytest, y_hat)))
print(y_hat)
print(ytest)

plt.plot(fpr, tpr)
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC curve for Logistic regression on EEG spectra")
plt.savefig("lr_roc.png")
plt.close()
