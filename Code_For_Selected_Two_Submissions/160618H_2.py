import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


training = pd.read_csv('train.csv')
testing = pd.read_csv('test.csv')

#print(training.head())
#print(testing.head())

#cleaning

#print(training.isna().sum())
#print(testing.isna().sum())

#fill missing values by mean of the column
training.fillna(training.mean(), inplace = True)
#print(training.isna().sum())

#dropping unnecessary colums
training.drop(['pickup_time','drop_time'], axis = 1, inplace = True)
testing.drop(['pickup_time', 'drop_time'], axis = 1, inplace = True)

#converting correct, incorrect to 1 and 0
training.label = training.label.map(dict(correct = 1, incorrect = 0))

#feature engineering
training_length = len(training)
dataset = pd.concat(objs = [training,testing], axis = 0).reset_index(drop = True)

#feature preprocessing

#scaling
scaler = MinMaxScaler(feature_range = (0,1))
features = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare', 'meter_waiting_till_pickup', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'fare']
dataset[features] = scaler.fit_transform(dataset[features])

#splitting to training and testing
training = dataset[:training_length]
testing = dataset[training_length:]
testing.drop(labels = ['label'], axis = 1 , inplace = True)

training['label'] = training['label'].astype(int)

training.drop(labels = ['tripid'], axis = 1, inplace = True)

Y = training['label']
X = training.drop(labels = ['label'], axis = 1)

#extracting top 10 best features

best = SelectKBest(score_func=chi2, k=10)
fit = best.fit(X, Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes
scores = pd.concat([dfcolumns, dfscores], axis=1)
scores.columns = ['Feature', 'Score']  
#print(scores.nlargest(30, 'Score'))  


#trainign and testing

y = training['label']
x = training.drop(labels = 'label', axis = 1)

'''
kfold = StratifiedKFold(n_splits = 20, random_state = None , shuffle=False)

random_state = None

classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers:
    cv_results.append(
        cross_val_score(classifier, x, y=y, scoring="f1", cv=kfold, n_jobs=4)
    )

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

print(cv_res)

'''

model = ExtraTreesClassifier()
model.fit(x,y)

trip_ids = testing.tripid
test = testing.drop(labels = ['tripid'], axis = 1)
predictions = model.predict(test)

output = pd.DataFrame({'tripid': trip_ids , 'prediction': predictions})
output.to_csv('160618H_submission2.csv', index = False)
print('Done!!!!!!!!!!!!')
