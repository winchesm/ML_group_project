import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as Vectorizer
import matplotlib.pyplot as plt
import scipy.sparse
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report as metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

from sklearn.preprocessing import StandardScaler

def category_index_map(categories):
    out_vec = np.zeros(6, int)
    if "echnology" in categories:
        out_vec[0] = 1
    if "ntertainment" in categories:
        out_vec[1] = 1
    if "esign" in categories:
        out_vec[2] = 1
    if "usiness" in categories:
        out_vec[3] = 1
    if "cience" in categories:
        out_vec[4] = 1
    if "lobal issues" in categories:
        out_vec[5] = 1
    return out_vec

def prepare_output(v):
    out_vec = np.zeros((len(v), 6), int)
    for i in range(len(v)):
        out_vec[i] = category_index_map(v[i])
    return out_vec




df = pd.read_csv("data.csv")

y = prepare_output(df.iloc[:,1])

data_matrix = scipy.sparse.load_npz('processed_data.npz')

sc_X = StandardScaler(with_mean=False)

X_train, X_test, y_train, y_test = train_test_split(data_matrix, y, test_size=0.20, random_state=42)

X_trainscaled = sc_X.fit_transform(X_train)
X_testscaled = sc_X.transform(X_test)

mean_error = []
std_error = []

size_range = [500, 5000, 10000, 50000]
Ci_range = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 100]

fig = plt.figure()
ax = fig.add_subplot(title= "Plot of mse vs value of C")
ax.set_xlabel('C')
ax.set_ylabel('mean square error')
for n in size_range:
    for Ci in Ci_range:
        model_mlp = MLPClassifier(activation ='logistic',hidden_layer_sizes=(n, round(n/2), round(n/4)),  max_iter=50,  alpha = 1.0/Ci)

        scores = cross_val_score(model_mlp, X_trainscaled, y_train, cv=5, scoring='neg_mean_squared_error')
        mean_error.append(-scores.mean())
        std_error.append(-scores.std())
    labl = ' size of first deep layer' + str(n)#this change to size
    plt.errorbar(Ci_range, mean_error, yerr=std_error, label=labl, linewidth=2)
    print(mean_error)
    print(std_error)
    mean_error.clear()
    std_error.clear()


n = 5000
Ci = 100
model_mlp = MLPClassifier(activation = 'logistic' ,hidden_layer_sizes=(n, round(n/2), round(n/4)),  max_iter=50,  alpha = 1.0/Ci).fit( X_trainscaled, y_train)
y_pred_s = model_mlp.predict(X_testscaled )
print(metrics(y_test, y_pred_s))