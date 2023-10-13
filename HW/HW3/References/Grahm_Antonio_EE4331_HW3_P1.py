#************************************************************************************
# Antonio Grahm
# ML – HW#3
# Filename: Grahm_Antonio_EE4331_HW3_P1.py
# Due: Oct. 6, 2023
#
# Objective:
#
#*************************************************************************************

# Importing the required libraries
import sys
import warnings
import matplotlib
import numpy as np
import pandas as pd
from tkinter import W
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score,auc,  \
                            f1_score, precision_score,roc_curve, roc_auc_score

# Suppress future warnings because what is backward-compatability.
warnings.simplefilter(action='ignore',category=FutureWarning)

#################################
# PLOTTING DECISION CODE  - pg 32
#################################

def plotDecisionRegions(name,X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
        y=X[y == cl, 1],
        alpha=0.8,
        c=colors[idx],
        marker=markers[idx],
        label=cl,
        edgecolor='black')

    #plot with labels and legend
    plt.title("{} Decision Regions".format(name))
    plt.legend()

    #Save plot ---> LEAP requires this part
    plt.savefig("{}_Decision_Regions.png".format(name))
    plt.close()

##########################
# End of PLOTTING DECISION CODE 
##########################

####################################
# PLOT LEARNING CURVE CODE - pg 197
####################################

def plotLearningCurve(pipeline, X, y, name,train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure(figsize=(8, 6))
    plt.title("Learning Curve")

    plt.xlabel("Number of Training Samples")
    plt.ylabel("Accuracy")
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X, y, cv=10, n_jobs=1, train_sizes=train_sizes)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    test_mean  = np.mean(test_scores, axis=1)
    test_std   = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.grid()
    plt.legend(loc="lower right")
    plt.savefig('{}_Learning_Curve.png'.format(name))
    #plt.show()
    plt.close()

####################################
# End of PLOT LEARNING CURVE CODE
####################################

####################################
# PLOT CONFUSION MATRIX CODE - pg 206
####################################
def plotConfusion(y_pred_train,y_train,y_test,y_pred,name):

    train_confusion = confusion_matrix(y_train,y_pred_train)
    test_confusion  = confusion_matrix(y_test,y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.matshow(train_confusion,cmap=matplotlib.colormaps['Blues'],alpha=0.3)
    for i in range(train_confusion.shape[0]):
        for j in range(train_confusion.shape[1]):
            plt.text(x=j, y=i,
            s=train_confusion[i, j],va='center',ha='center')

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Training Data")
    plt.savefig('{}_Train_Confusion_Matrix.png'.format(name))
    #plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.matshow(test_confusion,cmap=matplotlib.colormaps['Blues'],alpha=0.3)
    for i in range(test_confusion.shape[0]):
        for j in range(test_confusion.shape[1]):
            plt.text(x=j, y=i,
            s=test_confusion[i, j],va='center',ha='center')

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Testing Data")
    plt.savefig('{}_Test_Confusion_Matrix.png'.format(name))
    #plt.show()
    plt.close()

####################################
# End of PLOT CONFUSION MATRIX CODE
####################################
    
####################################
# PLOT ROC_AUC CODE
####################################
def plotROCAUC(clf,X_test,y_test,name):

    y_prob    = clf.predict_proba(X_test)
    X_targets = len(np.unique(y_test))

    fpr     = dict()
    tpr     = dict()
    roc_auc = dict()

    for i in range(X_targets):
        fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(X_targets):
        plt.plot(fpr[i], tpr[i], lw=2,
                label='ROC (Area = {:.2f}) for class {}'.format(roc_auc[i],i))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    plt.title('{}: ROC_AUC'.format(name))
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.savefig('{}_ROC_AUC.png'.format(name))
    plt.close()
####################################
# End of PLOT ROC_AUC CODE
####################################

####################################
# BEST LR PARAMETERS CODE
####################################

def lrBestParams():

    df = pd.read_csv('Loc_0202_Master.csv')

    X = df[['freq[Hz]', 're:Trc1_S11', 'im:Trc1_S11', 're:Trc2_S21','im:Trc2_S21']].to_numpy()
    y = LabelEncoder().fit_transform(df["room"])

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)

    lr_param_grid_pca = {
        'dim_reduction__n_components' : [3,4,5],
        'clf__penalty'                : ['l2'],
        'clf__C'                      : [0.01,0.1,1,10,100],
        'clf__solver'                 : ['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga'],
        'clf__max_iter'               : [500,1000,1500],
        'clf__multi_class'            : ['auto','ovr']
        }

    lr_param_grid_lda = {
        'dim_reduction__n_components' : [2,3],
        'clf__penalty'                : ['l2'],
        'clf__C'                      : [0.01,0.1,1,10,100],
        'clf__solver'                 : ['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga'],
        'clf__max_iter'               : [500,1000,1500],
        'clf__multi_class'            : ['auto','ovr']
        }

    # Standardized/PCA
    gs_best = None
    best_score = 0

    pipe_lr = Pipeline([
                ('scaler', StandardScaler()),
                ('dim_reduction', PCA()), 
                ('clf', LogisticRegression())])

    gs = GridSearchCV(pipe_lr,
                    lr_param_grid_pca,
                    cv = 10)

    gs = gs.fit(X_train,y_train)

    if gs.best_score_ > best_score:
        gs_best = gs

    # Standardized/LDA
    pipe_lr = Pipeline([
                ('scaler'       , StandardScaler()),
                ('dim_reduction', LDA()), 
                ('clf'          , LogisticRegression())])

    gs = GridSearchCV(pipe_lr,
                    lr_param_grid_lda,
                    cv=10)

    gs = gs.fit(X_train,y_train)

    if gs.best_score_ > best_score:
        gs_best = gs

    # Normalized/PCA
    pipe_lr = Pipeline([
                ('scaler'       , MinMaxScaler()),
                ('dim_reduction', PCA()), 
                ('clf'          , LogisticRegression())])

    gs = GridSearchCV(pipe_lr,
                    lr_param_grid_pca,
                    cv=10)

    gs = gs.fit(X_train,y_train)

    if gs.best_score_ > best_score:
        gs_best = gs

    #Normalized/LDA
    pipe_lr = Pipeline([
                ('scaler'       , MinMaxScaler()),
                ('dim_reduction', LDA()), 
                ('clf'          , LogisticRegression())])

    gs = GridSearchCV(pipe_lr,
                    lr_param_grid_lda,
                    cv=10)

    gs = gs.fit(X_train,y_train)

    if gs.best_score_ > best_score:
        gs_best = gs


    # Test model with best found parameters
    y_pred       = gs_best.predict(X_test)
    y_pred_train = gs_best.predict(X_train)

    # Gather scoring metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')
    f1        = f1_score(y_test, y_pred, average='weighted')

    # Log Results to text file
    with open("EE4331_HW3_Best_Performing_Models.txt", 'w') as f:

        f.write('*-------------- Highest Accuracy Parameters for LOGISTIC REGRESSION MODEL ------------*\n')
        f.write('Data Split: 80/20\n')
        f.write('Best Parameters\n{}\n'.format(gs_best.best_params_))
        f.write('Training Accuracy {:.2f}\n'.format(train_acc))
        f.write('Testing Accuracy {:.2f}\n'.format(test_acc))
        f.write('Precision Score {:.2f}\n'.format(precision))
        f.write('Recall Score {:.2f}\n'.format(recall))
        f.write('F1 Score {:.2f}\n'.format(f1))
        f.write('*-------------------------------------------------------------------------------------*\n')

    plotLearningCurve(gs_best.best_estimator_, X_train, y_train,"Logistic_Regression", \
                      train_sizes=np.linspace(.1, 1.0, 5))
    plotConfusion(y_pred_train,y_train,y_pred,y_test, "Logistic_Regression")
    plotROCAUC(gs_best.best_estimator_,X_test,y_test,"Logistic Regression")

    #################################################
    # Reduce Dimensions for plotting decision regions
    #################################################

    # If PCA was used for best model
    if type(gs_best.best_estimator_['dim_reduction']) == PCA:

        # If Standardization was used
        if type(gs_best.best_estimator_['scaler'] == StandardScaler()):

            sc = StandardScaler()
            pca = PCA(n_components=2)

            X_train_std = sc.fit_transform(X_train)
            X_test_std  = sc.fit_transform(X_test)

            X_train_final = pca.fit_transform(X_train_std)
            X_test_final  = pca.fit_transform(X_test_std)

            gs_plot = Pipeline([
                        ('scaler'       , StandardScaler()),
                        ('dim_reduction', PCA(n_components=2)), 
                        ('clf'          , LogisticRegression(
                                            C           = gs_best.best_params_['clf__C'],
                                            max_iter    = gs_best.best_params_['clf__max_iter'],
                                            multi_class = gs_best.best_params_['clf__multi_class'],
                                            solver      = gs_best.best_params_['clf__solver'],
                                            penalty     = 'l2'
                                            ))])  
        # If normalization was used
        else:

            mm = MinMaxScaler()
            pca = PCA(n_components=2)

            X_train_norm = mm.fit_transform(X_train)
            X_test_norm  = mm.fit_transform(X_test)

            X_train_final = pca.fit_transform(X_train_norm)
            X_test_final  = pca.fit_transform(X_test_norm)            

            gs_plot = Pipeline([
                        ('scaler'       , MinMaxScaler()),
                        ('dim_reduction', PCA(n_components=2)), 
                        ('clf'          , LogisticRegression(
                                            C           = gs_best.best_params_['clf__C'],
                                            max_iter    = gs_best.best_params_['clf__max_iter'],
                                            multi_class = gs_best.best_params_['clf__multi_class'],
                                            solver      = gs_best.best_params_['clf__solver'],
                                            penalty     = 'l2'
                                            ))])  
    # Used LDA
    else:
        
        # If Standardization was used
        if type(gs_best.best_estimator_['scaler'] == StandardScaler()):

            sc = StandardScaler()
            pca = LDA(n_components=2)

            X_train_std = sc.fit_transform(X_train,y_train)
            X_test_std  = sc.fit_transform(X_test,y_test)

            X_train_final = pca.fit_transform(X_train_std,y_train)
            X_test_final  = pca.fit_transform(X_test_std,y_test)

            # Create new pipeline with best parameters and reduced components.
            gs_plot = Pipeline([
                        ('scaler'       , StandardScaler()),
                        ('dim_reduction', LDA(n_components=2)), 
                        ('clf'          , LogisticRegression(
                                            C           = gs_best.best_params_['clf__C'],
                                            max_iter    = gs_best.best_params_['clf__max_iter'],
                                            multi_class = gs_best.best_params_['clf__multi_class'],
                                            solver      = gs_best.best_params_['clf__solver'],
                                            penalty     = 'l2'
                                            ))])  

        # If Normalization was used
        else:
            mm = MinMaxScaler()
            pca = LDA(n_components=2)

            X_train_norm = mm.fit_transform(X_train,y_train)
            X_test_norm  = mm.fit_transform(X_test,y_test)

            X_train_final = pca.fit_transform(X_train_norm,y_train)
            X_test_final  = pca.fit_transform(X_test_norm,y_test) 

            # Create new pipeline with best parameters and reduced components.
            gs_plot = Pipeline([
                        ('scaler'       , MinMaxScaler()),
                        ('dim_reduction', LDA(n_components=2)), 
                        ('clf'          , LogisticRegression(
                                            C           = gs_best.best_params_['clf__C'],
                                            max_iter    = gs_best.best_params_['clf__max_iter'],
                                            multi_class = gs_best.best_params_['clf__multi_class'],
                                            solver      = gs_best.best_params_['clf__solver'],
                                            penalty     = 'l2'
                                            ))])        

    # Prepare data to plot
    X_combined=np.vstack((X_train_final,X_test_final))
    y_combined=np.hstack((y_train,y_test))
    
    # Fit plotting model with reduced data
    gs_plot = gs_plot.fit(X_train_final,y_train)

    plotDecisionRegions("Logistic Regression", X=X_combined, y=y_combined, \
                          classifier=gs_plot)
    
####################################
# MAIN
####################################
def main():
    lrBestParams()
####################################
# End of MAIN
####################################

if __name__ == "__main__":
    sys.exit(main())