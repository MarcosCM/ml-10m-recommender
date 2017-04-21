#!/usr/bin/python
#In 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tag_headers = ['user_id', 'movie_id', 'tag', 'timestamp']
tags = pd.read_table('data/tags.dat', sep='::', header=None, names=tag_headers)

rating_headers = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('data/ratings.dat', sep='::', header=None, names=rating_headers)

movie_headers = ['movie_id', 'title', 'genres']
movies = pd.read_table('data/movies.dat', sep='::', header=None, names=movie_headers)
movie_titles = movies.title.tolist()

#In 2
movies.head()

#In 3
ratings.head()

#In 4
tags.head()

#In 5
df = movies.join(ratings, on=['movie_id'], rsuffix='_r').join(tags, on=['movie_id'], rsuffix='_t')
del df['movie_id_r']
del df['user_id_t']
del df['movie_id_t']
del df['timestamp_t']

#In 6
df.head()

#In 7
rp = df.pivot_table(columns=['movie_id'],index=['user_id'],values='rating')
rp.head()

#In 8
rp = rp.fillna(0); # Replace NaN
rp.head()

#In 9
Q = rp.values

#In 10
Q

#In 11
Q.shape

#In 12
W = Q>0.5
W[W == True] = 1
W[W == False] = 0
# To be consistent with our Q matrix
W = W.astype(np.float64, copy=False)

#In 13
W

#In 14
lambda_ = 0.1
n_factors = 100
m, n = Q.shape
n_iterations = 20

#In 15
X = 5 * np.random.rand(m, n_factors) 
Y = 5 * np.random.rand(n_factors, n)

#In 16
def get_error(Q, X, Y, W):
	return np.sum((W * (Q - np.dot(X, Y)))**2)

#In 17
errors = []
for ii in range(n_iterations):
	X = np.linalg.solve(np.dot(Y, Y.T) + lambda_ * np.eye(n_factors), np.dot(Y, Q.T)).T
	Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(n_factors), np.dot(X.T, Q))
	if ii % 100 == 0:
		print('{}th iteration is completed'.format(ii))
	errors.append(get_error(Q, X, Y, W))
Q_hat = np.dot(X, Y)
print('Error of rated movies: {}'.format(get_error(Q, X, Y, W)))

#In 18
plt.plot(errors);
plt.ylim([0, 20000]);

#In 20
weighted_errors = []
for ii in range(n_iterations):
    for u, Wu in enumerate(W):
        X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
                               np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
    for i, Wi in enumerate(W.T):
        Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
                                 np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
    weighted_errors.append(get_error(Q, X, Y, W))
    print('{}th iteration is completed'.format(ii))
weighted_Q_hat = np.dot(X,Y)
#print('Error of rated movies: {}'.format(get_error(Q, X, Y, W)))

#In 26
plt.plot(weighted_errors);
plt.xlabel('Iteration Number');
plt.ylabel('Mean Squared Error');

#In 28
def print_recommendations(W=W, Q=Q, Q_hat=Q_hat, movie_titles=movie_titles):
	#Q_hat -= np.min(Q_hat)
	#Q_hat[Q_hat < 1] *= 5
	Q_hat -= np.min(Q_hat)
	Q_hat *= float(5) / np.max(Q_hat)
	movie_ids = np.argmax(Q_hat - 5 * W, axis=1)
	for jj, movie_id in zip(range(m), movie_ids):
		#if Q_hat[jj, movie_id] < 0.1: continue
		print('User {} liked {}\n'.format(jj + 1, ', '.join([movie_titles[ii] for ii, qq in enumerate(Q[jj]) if qq > 3])))
		print('User {} did not like {}\n'.format(jj + 1, ', '.join([movie_titles[ii] for ii, qq in enumerate(Q[jj]) if qq < 3 and qq != 0])))
		print('\n User {} recommended movie is {} - with predicted rating: {}'.format(jj + 1, movie_titles[movie_id], Q_hat[jj, movie_id]))
		print('\n' + 100 *  '-' + '\n')
#print_recommendations()

#In 29
print_recommendations(Q_hat=weighted_Q_hat)