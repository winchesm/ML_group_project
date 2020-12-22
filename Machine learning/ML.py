import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as Vectorizer
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer as Stemmer
from nltk.stem import WordNetLemmatizer
import scipy.sparse
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import RidgeClassifier as Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report as metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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
    out_vec = np.zeros((len(v), 6), int)
    for i in range(len(v)):
        out_vec[i] = category_index_map(v[i])
    return out_vec


def prepare_input_l(v):
    # stemmer = Stemmer("english", ignore_stopwords=True)
    lemmatizer = WordNetLemmatizer()
    in_vec = ["" for x in range(len(v))]
    vectorizer = Vectorizer(input="content", stop_words='english')
    tokenizer = vectorizer.build_tokenizer()
    for i in range(len(v)):
        tokens = tokenizer(v[i].lower())
        # stems = [stemmer.stem(token) for token in tokens]
        stems = [lemmatizer.lemmatize(token) for token in tokens]
        sentence = " ".join(stems)
        in_vec[i] = sentence
    tmp = vectorizer.fit_transform(in_vec)
    print(vectorizer.get_feature_names())
    return tmp


def prepare_input_s(v):
    stemmer = Stemmer()
    in_vec = ["" for x in range(len(v))]
    vectorizer = Vectorizer(input="content", stop_words='english')
    tokenizer = vectorizer.build_tokenizer()
    for i in range(len(v)):
        tokens = tokenizer(v[i].lower())
        stems = [stemmer.stem(token) for token in tokens]
        sentence = " ".join(stems)
        in_vec[i] = sentence
    tmp = vectorizer.fit_transform(in_vec)
    print(vectorizer.get_feature_names())
    return tmp


def prepare_input_d2v(v):
    stemmer = Stemmer()
    in_vec = ["" for x in range(len(v))]
    vectorizer = Vectorizer(input="content", stop_words='english')
    tokenizer = vectorizer.build_tokenizer()
    '''for i in range(len(v)):
        tokens = tokenizer(v[i].lower())
        stems = [stemmer.stem(token) for token in tokens]
        sentence = " ".join(stems)
        in_vec[i] = sentence
    tmp = vectorizer.fit_transform(in_vec)'''
    # tokenss = [stemmer.stem(tokens) for tokens in [tokenizer(_d.lower()) for _, _d in enumerate(v)]]
    # tokenss = [stemmer.stem(tokenizer(_d)) for _, _d in enumerate(v)]
    tokenss = [[stemmer.stem(_dd) for _dd in tokenizer(_d)] for _, _d in enumerate(v)]
    tagged_data = [TaggedDocument(words=tokenss[i], tags=[str(i)]) for i, _d in enumerate(v)]
    model = Doc2Vec(tagged_data, vector_size=120, window=10, min_count=2, workers=4, epochs=500)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    a = [model.infer_vector(tokens) for tokens in tokenss]
    model.save("doc2vec.model")
    return a


def run_knn_s(ks):
    y_nums = sum(y)
    means = np.zeros(len(ks))
    stds = np.zeros(len(ks))

    for i in range(len(ks)):
        k = ks[i]
        kFold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
        model_ridge = KNN(n_neighbors=k).fit(X_train_s, y_train_s)
        scores_tmp = cross_val_score(model_ridge, X_test_s, y_test_s, cv=kFold, scoring='neg_mean_squared_error')
        means[i] -= np.mean(scores_tmp)
        stds[i] += np.std(scores_tmp)
    plt.figure()
    plt.errorbar(ks, means, yerr=stds)
    return -means


def run_knn_l(ks):
    y_nums = sum(y)
    means = np.zeros(len(ks))
    stds = np.zeros(len(ks))

    for i in range(len(ks)):
        k = ks[i]
        kFold = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
        model_ridge = KNN(n_neighbors=k).fit(X_train_l, y_train_l)
        scores_tmp = cross_val_score(model_ridge, X_test_l, y_test_l, cv=kFold, scoring='neg_mean_squared_error')
        means[i] -= np.mean(scores_tmp)
        stds[i] += np.std(scores_tmp)
    plt.figure()
    plt.errorbar(ks, means, yerr=stds)
    return -means


def run_ridge_s(alphas):
    y_nums = sum(y)
    means = np.zeros(len(alphas))
    stds = np.zeros(len(alphas))

    for i in range(len(alphas)):
        alpha = alphas[i]
        kFold = RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)
        for j in range(6):
            model_knn = Ridge(alpha=alpha).fit(X_train_s, y_train_s[:, j])
            scores_tmp = cross_val_score(model_knn, X_test_s, y_test_s[:, j], cv=kFold, scoring='neg_mean_squared_error')
            means[i] -= y_nums[j]*np.mean(scores_tmp)/sum(y_nums)
            stds[i] += y_nums[j]*np.std(scores_tmp)/sum(y_nums)
    plt.figure()
    plt.errorbar(alphas, means, yerr=stds)
    return -means


def run_ridge_l(alphas):
    y_nums = sum(y)
    means = np.zeros(len(alphas))
    stds = np.zeros(len(alphas))

    for i in range(len(alphas)):
        alpha = alphas[i]
        kFold = RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)
        for j in range(6):
            model_knn = Ridge(alpha=alpha).fit(X_train_l, y_train_l[:, j])
            scores_tmp = cross_val_score(model_knn, X_test_l, y_test_l[:, j], cv=kFold, scoring='neg_mean_squared_error')
            means[i] -= y_nums[j]*np.mean(scores_tmp)/sum(y_nums)
            stds[i] += y_nums[j]*np.std(scores_tmp)/sum(y_nums)
    plt.figure()
    plt.errorbar(alphas, means, yerr=stds)
    return -means


df = pd.read_csv("data.csv")
X = prepare_input_s(np.array(df.iloc[:, 2]).astype(str))
X2 = prepare_input_l(np.array(df.iloc[:, 2]).astype(str))
y = prepare_output(df.iloc[:, 1])

scipy.sparse.save_npz('processed_data.npz', X)
scipy.sparse.save_npz('processed_data_lemmatized.npz', X2)
data_matrix = scipy.sparse.load_npz('processed_data.npz')
data_matrix_lemmatized = scipy.sparse.load_npz('processed_data_lemmatized.npz')

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(data_matrix, y, test_size=0.20, random_state=42)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(data_matrix_lemmatized, y, test_size=0.20, random_state=42)
knn_errors_s = run_knn_s((1, 5, 10, 25, 50, 100, 150, 250, 400))
knn_errors_l = run_knn_l((1, 5, 10, 25, 50, 100, 150, 250, 400))
ridge_errors_s = run_ridge_s((0.00001, 0.01, 0.1, 1, 10))
ridge_errors_l = run_ridge_l((0.00001, 0.01, 0.1, 1, 10))
# print(knn_errors)
# print(ridge_errors)
print("Stemmed")
print("Ridge")
model_ridge_0 = Ridge(alpha=0.1).fit(X_train_s, y_train_s[:, 0])
model_ridge_1 = Ridge(alpha=0.1).fit(X_train_s, y_train_s[:, 1])
model_ridge_2 = Ridge(alpha=0.1).fit(X_train_s, y_train_s[:, 2])
model_ridge_3 = Ridge(alpha=0.1).fit(X_train_s, y_train_s[:, 3])
model_ridge_4 = Ridge(alpha=0.1).fit(X_train_s, y_train_s[:, 4])
model_ridge_5 = Ridge(alpha=0.1).fit(X_train_s, y_train_s[:, 5])
y_pred_s = np.column_stack((model_ridge_0.predict(X_test_s), model_ridge_1.predict(X_test_s), model_ridge_2.predict(X_test_s), model_ridge_3.predict(X_test_s), model_ridge_4.predict(X_test_s), model_ridge_5.predict(X_test_s)))
multilabel_confusion_matrix(y_test_s, y_pred_s)
print(metrics(y_test_s, y_pred_s))
print("KNN")
model_knn = KNN(n_neighbors=10).fit(X_train_s, y_train_s)
y_pred_s = model_knn.predict(X_test_s)
multilabel_confusion_matrix(y_test_s, y_pred_s)
print(metrics(y_test_s, y_pred_s))
print("Lemmatized")
print("Ridge")
model_ridge_0 = Ridge(alpha=0.1).fit(X_train_l, y_train_l[:, 0])
model_ridge_1 = Ridge(alpha=0.1).fit(X_train_l, y_train_l[:, 1])
model_ridge_2 = Ridge(alpha=0.1).fit(X_train_l, y_train_l[:, 2])
model_ridge_3 = Ridge(alpha=0.1).fit(X_train_l, y_train_l[:, 3])
model_ridge_4 = Ridge(alpha=0.1).fit(X_train_l, y_train_l[:, 4])
model_ridge_5 = Ridge(alpha=0.1).fit(X_train_l, y_train_l[:, 5])
y_pred_l = np.column_stack((model_ridge_0.predict(X_test_l), model_ridge_1.predict(X_test_l), model_ridge_2.predict(X_test_l), model_ridge_3.predict(X_test_l), model_ridge_4.predict(X_test_l), model_ridge_5.predict(X_test_l)))
multilabel_confusion_matrix(y_test_l, y_pred_l)
print(metrics(y_test_l, y_pred_l))
print("KNN")
model_knn = KNN(n_neighbors=10).fit(X_train_l, y_train_l)
y_pred_l = model_knn.predict(X_test_l)
multilabel_confusion_matrix(y_test_l, y_pred_l)
print(metrics(y_test_l, y_pred_l))

plt.show()
