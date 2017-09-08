
# coding: utf-8

# In[1]:

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd


# In[ ]:




# In[2]:

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[3]:

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[4]:

train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")


# In[5]:

combine = [train, test]


# In[6]:

print(train.columns.values)


# In[7]:

train.head()


# In[8]:

train.tail()


# In[9]:

train.info()
print("_" * 40)
test.info()


# In[10]:

train.describe()


# In[11]:

train.describe(include=['O'])


# In[12]:

train[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived", ascending=False)


# In[13]:

train[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Survived", ascending=False)


# In[14]:

train[["SibSp", "Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(by="Survived", ascending=False)


# In[15]:

train[["Parch", "Survived"]].groupby(["Parch"], as_index=False).mean().sort_values(by="Survived", ascending=False)


# In[16]:

g = sns.FacetGrid(train, col="Survived")
g.map(plt.hist, "Age", bins=20)


# In[17]:

grid = sns.FacetGrid(train, row="Embarked", size=2.2, aspect=1.6)
grid.map(sns.pointplot, "Pclass", "Survived", "Sex", palette="deep")
grid.add_legend()


# In[18]:

grid = sns.FacetGrid(train, row="Embarked", col="Survived", size=2.2, aspect=1.6)
grid.map(sns.barplot, "Sex", "Fare", alpha=0.5, ci=None)
grid.add_legend()


# In[19]:

print("Before", train.shape, test.shape, combine[0].shape, combine[1].shape)

train = train.drop(["Ticket", "Cabin"], axis=1)
test = test.drop(["Ticket", "Cabin"], axis=1)
combine=[train, test]

print("After", train.shape, test.shape, combine[0].shape, combine[1].shape)


# In[20]:

for dataset in combine:
    dataset["Title"] = dataset.Name.str.extract(" ([A-Za-z]+)\.", expand=False)

pd.crosstab(train["Title"], train["Sex"])


# In[21]:

for dataset in combine:
    dataset["Title"] = dataset["Title"].replace(['Lady', 'Countess','Capt', 'Col',        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset["Title"] = dataset["Title"].replace("Mlle", "Miss")
    dataset["Title"] = dataset["Title"].replace("Ms", "Miss")
    dataset["Title"] = dataset["Title"].replace("Mme", "Mrs")

train[["Title", "Survived"]].groupby(["Title"], as_index=False).mean()


# In[22]:

title_mapping  ={"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset["Title"] = dataset["Title"].map(title_mapping)
    dataset["Title"] = dataset["Title"].fillna(0)

train.head()


# In[23]:

train = train.drop(["Name", "PassengerId"], axis=1)
test = test.drop(["Name"], axis=1)
combine = [train, test]
train.shape, test.shape


# In[24]:

for dataset in combine:
    dataset["Sex"] = dataset["Sex"].map({"female": 1, "male": 0}).astype(int)
    
train.head()


# In[25]:

grid = sns.FacetGrid(train, row="Pclass", col="Sex", size=2.2, aspect=1.6)
grid.map(plt.hist, "Age", alpha=0.5, bins=20)
grid.add_legend()


# In[26]:

guess_ages = np.zeros((2,3))
guess_ages


# In[27]:

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset["Sex"] == i) &                               (dataset["Pclass"] == j+1)]["Age"].dropna()
            age_guess = guess_df.median()
            
            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess/0.5 + 0.5) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                        "Age"] = guess_ages[i, j]
    
    dataset["Age"] = dataset["Age"].astype(int)

train.head()


# In[28]:

# Create age bands and determine correlations with Survived
train["AgeBand"] = pd.cut(train["Age"], 5)
train[["AgeBand", "Survived"]].groupby(["AgeBand"], as_index=False).mean().sort_values(by="AgeBand", ascending=True)


# In[29]:

# Replace age with ordinals based on the bands
for dataset in combine:
    dataset.loc[dataset["Age"] <= 16, "Age"] = 0
    dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <= 32), "Age"] = 1
    dataset.loc[(dataset["Age"] > 32) & (dataset["Age"] <= 48), "Age"] = 2
    dataset.loc[(dataset["Age"] > 48) & (dataset["Age"] <= 64), "Age"] = 3
    dataset.loc[dataset["Age"] > 64, "Age"]

train.head()


# In[30]:

# AgeBand can now be removed
train = train.drop(["AgeBand"], axis=1)
combine = [train, test]
train.head()


# In[31]:

# Create FamilySize, which combines Parch and SibSp
for dataset in combine:
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1
    
train[["FamilySize", "Survived"]].groupby(["FamilySize"], as_index=False).mean().sort_values(by="Survived", ascending=False)


# In[32]:

# Create another feature called IsAlone
for dataset in combine:
    dataset["IsAlone"] = 0
    dataset.loc[dataset["FamilySize"] == 1, "IsAlone"] = 1
    
train[["IsAlone", "Survived"]].groupby(["IsAlone"], as_index=False).mean()


# In[33]:

# Drop Parch, SibSp, and FamilySize in favor of IsAlone
train = train.drop(["Parch", "SibSp", "FamilySize"], axis=1)
test = test.drop(["Parch", "SibSp", "FamilySize"], axis=1)
combine=[train, test]

train.head()


# In[34]:

# Create a feature combining Pclass and Age
for dataset in combine:
    dataset["Age*Class"] = dataset.Age * dataset.Pclass
    
train.loc[:, ["Age*Class", "Age", "Pclass"]].head(10)


# In[35]:

# Training dataset has two missing values for port of embarkation. These can be filled with the most common occurance.
freq_port = train.Embarked.dropna().mode()[0]
freq_port


# In[36]:

for dataset in combine:
    dataset["Embarked"] = dataset["Embarked"].fillna(freq_port)
    
train[["Embarked", "Survived"]].groupby(["Embarked"], as_index=False).mean().sort_values(by="Survived", ascending=False)


# In[37]:

# Convert Embarked to a new numeric Port feature
for dataset in combine:
    dataset["Embarked"] = dataset["Embarked"].map( {"S": 0, "C": 1, "Q": 2} ).astype(int)
    
train.head()


# In[38]:

# Single missing value in Fare in test. Replace the median Fare and since only one value, can do in a single line of code
test["Fare"].fillna(test["Fare"].dropna().median(), inplace=True)
test.head()


# In[39]:

# Create FareBand
train["FareBand"] = pd.qcut(train["Fare"], 4)
train[["FareBand", "Survived"]].groupby(["FareBand"], as_index=False).mean().sort_values(by="Survived", ascending=False)


# In[40]:

# Convert Fare to ordinal values based on FareBand
for dataset in combine:
    dataset.loc[dataset["Fare"] <= 7.91, "Fare"] = 0
    dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 14.454), "Fare"] = 1
    dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 31), "Fare"] = 2
    dataset.loc[dataset["Fare"] > 31, "Fare"] = 3
    dataset["Fare"] = dataset["Fare"].astype(int)

train = train.drop(["FareBand"], axis=1)
combine = [train, test]

train.head()


# In[41]:

test.head()


# In[42]:

# Model, predict, and solve
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[43]:

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train)*100, 2)
acc_log


# In[44]:

coeff_df = pd.DataFrame(train.columns.delete(0))
coeff_df.columns = ["Feature"]
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by="Correlation", ascending=False)


# In[45]:

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train)*100, 2)
acc_svc


# In[46]:

# k-Nearest Neighbors (k-NN)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train)*100, 2)
acc_knn


# In[47]:

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train)*100, 2)
acc_gaussian


# In[48]:

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train)*100, 2)
acc_perceptron


# In[49]:

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train)*100, 2)
acc_linear_svc


# In[50]:

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train)*100, 2)
acc_sgd


# In[51]:

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train)*100, 2)
acc_decision_tree


# In[52]:

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train)*100, 2)
acc_random_forest


# In[53]:

# Model evaluation
# While both Decision Tree and Random Forest score the same, we choose to use
# Random Forest as they correct for decision trees' habit of overfitting to their training set.
models = pd.DataFrame({
    "Model": ["Support Vector Machines", "KNN", "Logistic Regression",
             "Random Forest", "Naive Bayes", "Perceptron",
             "Stochastic Gradient Descent", "Linear SVC",
             "Decision Tree",],
    "Score": [acc_svc, acc_knn, acc_log, acc_random_forest,
             acc_gaussian, acc_perceptron, acc_sgd,
             acc_linear_svc, acc_decision_tree,]
})
models.sort_values(by="Score", ascending=False)


# In[54]:

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": Y_pred
})


# In[55]:

submission.head()


# In[56]:

submission.to_csv("./output/submission.csv", index=False)

