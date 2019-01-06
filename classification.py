import AMLS_assignment as l2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import neural_network
from sklearn import preprocessing as pp
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report, confusion_matrix

def get_data():
    X, Y , names = l2.extract_features_labels()
    #Y = np.array([y, -(y - 1)]).T

    split = int(0.8*len(X))
    tr_X = X[:split]
    tr_Y = Y[:split]
    te_X = X[split:]
    te_Y = Y[split:]
    te_names = names[split:]

    print(tr_X.shape, tr_Y.shape)
    print(Y[:, 1])
    print(len(X))
    return tr_X, tr_Y, te_X, te_Y, te_names, X, Y
tr_X, tr_Y, te_X, te_Y, names, allX, allY = get_data()

def train_SVM(tr_X, tr_Y, te_X, te_Y):

    # Define the classification model
    svclassifier = svm.SVC(kernel = 'sigmoid', gamma='auto')

    #Fit model
    svclassifier.fit(tr_X, tr_Y)
    print("Model:\n", svclassifier)

    #Prediction
    y_pred = svclassifier.predict(te_X)
    score = svclassifier.score(te_X, te_Y)

    #Evaluation
    print(confusion_matrix(te_Y, y_pred))
    print(classification_report(te_Y, y_pred))

    return svclassifier, y_pred, score


svcmodel, y_pred, accuracy = train_SVM(tr_X, tr_Y[:,4], te_X, te_Y[:,4])


# Storing to .csv
f = open(("humanSVM_4.csv"), "w+")

# Average inference accuracy
f.write("%0.3f \r\n" % accuracy)

# Predictions

print(len(names) , len(y_pred))

[f.write("%s, %d\r\n" % (str(int(names[i]))+'.png', y_pred[i])) for i in range(len(names))]
f.close()


# Scale landmarks with sklearn
# preprocessing fit to training data
def scale_dat(training_img, test_img):
    scaler = pp.StandardScaler()
    scaler.fit(training_img)
    tr_X = scaler.transform(training_img)
    te_X = scaler.transform(test_img)

    return tr_X, te_X

def train_MLP(tr_X, tr_Y, te_X, te_Y):
    # Scale (x,y) coordinates
    tr_X, te_X, = scale_dat(tr_X, te_X)

    # Define the classification model
    model = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 2), random_state=1)

    # Fit model
    model.fit(tr_X, tr_Y)
    print("Model:\n", model)

    # Prediction
    y_pred = model.predict(te_X)
    score = model.score(te_X, te_Y)

    # Evaluation
    print(confusion_matrix(te_Y, y_pred))
    print(classification_report(te_Y, y_pred))

    return model, y_pred, score


mlpmodel, y_pred, accuracy = train_MLP(tr_X, tr_Y[:, 4], te_X, te_Y[:, 4])

# Storing to .csv
f = open(("humanMLP_4.csv"), "w+")

# Average inference accuracy
f.write("%0.3f \r\n" % accuracy)

# Predictions
[f.write("%s, %d\r\n" % (str(int(names[i]))+'.png', y_pred[i])) for i in range(len(names))]
f.close()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    return plt

print(len(allX), "\n", len(allY))
print(allX.shape, "\n", allY[:,1].shape)

title = "Learning Curves ()"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
plot_learning_curve(svcmodel, title, allX, allY[:, 1].ravel(), ylim=(0.7, 1.01), cv=10, n_jobs=4)

plt.show()
