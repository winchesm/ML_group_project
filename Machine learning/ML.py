import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as Vectorizer
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer as Stemmer
from nltk.stem import WordNetLemmatizer
import scipy.sparse
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as Logistic
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report as metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import multilabel_confusion_matrix
nltk.download('wordnet')

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
    # one-hot encode the the 6 tags i question
    out_vec = np.zeros((len(v), 6), int)
    # for each transcript
    for i in range(len(v)):
        out_vec[i] = category_index_map(v[i])
    return out_vec


def prepare_input_l(v):
    # lemmatize and tokenise transcripts
    lemmatizer = WordNetLemmatizer()
    in_vec = ["" for x in range(len(v))]
    vectorizer = Vectorizer(input="content", stop_words='english')
    tokenizer = vectorizer.build_tokenizer()
    # for each transcript
    for i in range(len(v)):
        tokens = tokenizer(v[i].lower())
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        sentence = " ".join(lemmas)
        in_vec[i] = sentence
    tmp = vectorizer.fit_transform(in_vec)
    print(vectorizer.get_feature_names())
    print(len(vectorizer.get_feature_names()))
    return tmp


def prepare_input_s(v):
    # stem and tokenise transcripts
    stemmer = Stemmer()
    in_vec = ["" for x in range(len(v))]
    vectorizer = Vectorizer(input="content", stop_words='english')
    tokenizer = vectorizer.build_tokenizer()
    # for each transcript
    for i in range(len(v)):
        tokens = tokenizer(v[i].lower())
        stems = [stemmer.stem(token) for token in tokens]
        sentence = " ".join(stems)
        in_vec[i] = sentence
    tmp = vectorizer.fit_transform(in_vec)
    print(vectorizer.get_feature_names())
    print(len(vectorizer.get_feature_names()))
    return tmp


def run_knn_s(ks):
    # errors
    means = np.zeros(len(ks))
    # standard deviation
    stds = np.zeros(len(ks))

    # for all k's
    for i in range(len(ks)):
        k = ks[i]
        kFold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
        model_ridge = KNN(n_neighbors=k).fit(X_train_s, y_train_s)
        # 5-fold cross validation
        scores_tmp = cross_val_score(model_ridge, X_test_s, y_test_s, cv=kFold,
                                     scoring='neg_mean_squared_error')
        # weighted average of errors
        means[i] -= np.mean(scores_tmp)
        stds[i] += np.std(scores_tmp)
    fig, ax = plt.subplots()
    ax.errorbar(ks, means, yerr=stds)
    ax.set_xlabel('K')
    ax.set_ylabel('MSE')
    return -means


def run_knn_l(ks):
    # errors
    means = np.zeros(len(ks))
    # standard deviation
    stds = np.zeros(len(ks))

    # for all k's
    for i in range(len(ks)):
        k = ks[i]
        kFold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
        model_ridge = KNN(n_neighbors=k).fit(X_train_l, y_train_l)
        # 5-fold cross validation
        scores_tmp = cross_val_score(model_ridge, X_test_l, y_test_l, cv=kFold,
                                     scoring='neg_mean_squared_error')
        # weighted average of errors
        means[i] -= np.mean(scores_tmp)
        stds[i] += np.std(scores_tmp)
    fig, ax = plt.subplots()
    ax.errorbar(ks, means, yerr=stds)
    ax.set_xlabel('K')
    ax.set_ylabel('MSE')
    return -means


def run_ridge_s(alphas):
    y_nums = sum(y)
    # errors
    means = np.zeros(len(alphas))
    # standard deviation
    stds = np.zeros(len(alphas))

    # for all alphas
    for i in range(len(alphas)):
        alpha = alphas[i]
        print(alpha)
        kFold = RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)
        # for all output tags
        for j in range(6):
            model_logistic = Logistic(C=alpha, solver='saga', max_iter=5000).fit(X_train_s, y_train_s[:, j])
            # 5-fold cross validation
            scores_tmp = cross_val_score(model_logistic, X_test_s, y_test_s[:, j],
                                         cv=kFold, scoring='neg_mean_squared_error')
            # weighted average of errors
            means[i] -= y_nums[j]*np.mean(scores_tmp)/sum(y_nums)
            stds[i] += y_nums[j]*np.std(scores_tmp)/sum(y_nums)
    fig, ax = plt.subplots()
    ax.errorbar(alphas, means, yerr=stds)
    ax.set_xlabel('C')
    ax.set_ylabel('MSE')
    return -means


def run_ridge_l(alphas):
    y_nums = sum(y)
    # errors
    means = np.zeros(len(alphas))
    # standard deviation
    stds = np.zeros(len(alphas))

    # for all alphas
    for i in range(len(alphas)):
        alpha = alphas[i]
        print(alpha)
        kFold = RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)
        # for all output tags
        for j in range(6):
            model_logistic = Logistic(C=alpha, solver='saga', max_iter=5000).fit(X_train_l, y_train_l[:, j])
            # 5-fold cross validation
            scores_tmp = cross_val_score(model_logistic, X_test_l, y_test_l[:, j],
                                         cv=kFold, scoring='neg_mean_squared_error')
            # weighted average of errors
            means[i] -= y_nums[j]*np.mean(scores_tmp)/sum(y_nums)
            stds[i] += y_nums[j]*np.std(scores_tmp)/sum(y_nums)
    fig, ax = plt.subplots()
    ax.errorbar(alphas, means, yerr=stds)
    ax.set_xlabel('C')
    ax.set_ylabel('MSE')
    return -means

# read in and sanatize input
df = pd.read_csv("data.csv")
df = df[df['Unnamed: 3'].isnull()]
df.dropna(subset=['Transcript'], inplace=True)
df.to_csv("data_sanitized.csv", index=False)
df = pd.read_csv("data_sanitized.csv")
# prepare X
X = prepare_input_s(np.array(df.iloc[:, 2]).astype(str))
scipy.sparse.save_npz('processed_data.npz', X)
X2 = prepare_input_l(np.array(df.iloc[:, 2]).astype(str))
scipy.sparse.save_npz('processed_data_lemmatized.npz', X2)
# prepare y
y = prepare_output(df.iloc[:, 1])

data_matrix = scipy.sparse.load_npz('processed_data.npz')
data_matrix_lemmatized = scipy.sparse.load_npz('processed_data_lemmatized.npz')

# split testing/training data
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(data_matrix, y, test_size=0.20, random_state=42)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(data_matrix_lemmatized, y, test_size=0.20, random_state=42)

# plot cross validation with a range of hyper-parameters
knn_errors_s = run_knn_s((1, 5, 10, 25, 50, 100))
knn_errors_l = run_knn_l((1, 5, 10, 25, 50, 100))
ridge_errors_s = run_ridge_s((0.01, 0.1, 1, 10, 100, 200))
ridge_errors_l = run_ridge_l((0.01, 0.1, 1, 10, 100, 200))

print("Stemmed")
print("Logistic")
# train best Logistic regression model with stemmed inputs
model_ridge_0 = Logistic(C=100, solver='saga', max_iter=5000).fit(X_train_s, y_train_s[:, 0])
model_ridge_1 = Logistic(C=100, solver='saga', max_iter=5000).fit(X_train_s, y_train_s[:, 1])
model_ridge_2 = Logistic(C=100, solver='saga', max_iter=5000).fit(X_train_s, y_train_s[:, 2])
model_ridge_3 = Logistic(C=100, solver='saga', max_iter=5000).fit(X_train_s, y_train_s[:, 3])
model_ridge_4 = Logistic(C=100, solver='saga', max_iter=5000).fit(X_train_s, y_train_s[:, 4])
model_ridge_5 = Logistic(C=100, solver='saga', max_iter=5000).fit(X_train_s, y_train_s[:, 5])
y_pred_s = np.column_stack((model_ridge_0.predict(X_test_s), model_ridge_1.predict(X_test_s),
                            model_ridge_2.predict(X_test_s), model_ridge_3.predict(X_test_s),
                            model_ridge_4.predict(X_test_s), model_ridge_5.predict(X_test_s)))
print(multilabel_confusion_matrix(y_test_s, y_pred_s))
print(metrics(y_test_s, y_pred_s))
print("KNN")
# train best kNN model with stemmed inputs
model_knn = KNN(n_neighbors=10).fit(X_train_s, y_train_s)
y_pred_s = model_knn.predict(X_test_s)
print(multilabel_confusion_matrix(y_test_s, y_pred_s))
print(metrics(y_test_s, y_pred_s))
print("Lemmatized")
print("Logistic")
# train best Logistic regression model with lemmatize inputs
model_ridge_0 = Logistic(C=100, solver='saga', max_iter=5000).fit(X_train_l, y_train_l[:, 0])
model_ridge_1 = Logistic(C=100, solver='saga', max_iter=5000).fit(X_train_l, y_train_l[:, 1])
model_ridge_2 = Logistic(C=100, solver='saga', max_iter=5000).fit(X_train_l, y_train_l[:, 2])
model_ridge_3 = Logistic(C=100, solver='saga', max_iter=5000).fit(X_train_l, y_train_l[:, 3])
model_ridge_4 = Logistic(C=100, solver='saga', max_iter=5000).fit(X_train_l, y_train_l[:, 4])
model_ridge_5 = Logistic(C=100, solver='saga', max_iter=5000).fit(X_train_l, y_train_l[:, 5])
y_pred_l = np.column_stack((model_ridge_0.predict(X_test_l), model_ridge_1.predict(X_test_l),
                            model_ridge_2.predict(X_test_l), model_ridge_3.predict(X_test_l),
                            model_ridge_4.predict(X_test_l), model_ridge_5.predict(X_test_l)))
print(multilabel_confusion_matrix(y_test_l, y_pred_l))
print(metrics(y_test_l, y_pred_l))
print("KNN")
# train best kNN model with lemmatize inputs
model_knn = KNN(n_neighbors=10).fit(X_train_l, y_train_l)
y_pred_l = model_knn.predict(X_test_l)
print(multilabel_confusion_matrix(y_test_l, y_pred_l))
print(metrics(y_test_l, y_pred_l))

plt.show()
