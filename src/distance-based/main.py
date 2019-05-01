import cv2
from joblib import Parallel, delayed
import numpy as np
from sklearn.model_selection import KFold

from rep import generate_hist_rep
from utils import plot_ranks

testing_case = 0


def get_signature_from_heatmap(hm):
    nr = hm.shape[0]
    nc = hm.shape[1]
    # print hm

    sig = np.zeros((nr*nc, 3), dtype=np.float32)
    for r in range(nr):
        for c in range(nc):
            idx = r * nc + c
            sig[idx, 0] = hm[r, c]
            sig[idx, 1] = r
            sig[idx, 2] = c

    sig[:, 0] /= np.sum(sig[:, 0])
    # print sig
    return sig


def classwise_emd_dist(vec, mv):
    g = int(np.sqrt(len(vec)))
    h1 = vec.reshape((g, g)).astype(np.float32)
    h2 = mv.reshape((g, g)).astype(np.float32)

    s1 = get_signature_from_heatmap(h1)
    s2 = get_signature_from_heatmap(h2)
    # s2[0] *= 1.0/27.0
    if testing_case == 2:
        s1[0] *= 0.5

    dis= cv2.EMD(s1, s2, cv2.DIST_L1)
    # dis = cv2.cv.CalcEMD2(cv2.cv.fromarray(s1), cv2.cv.fromarray(s2), cv2.cv.CV_DIST_L2)
    return dis[0]



def closest_class_rep(vec, class_mats):
    r = [classwise_emd_dist(vec, mv) for mv in class_mats]
    return np.min(r)


def find_rank(r):
    """ Find the ranking of the probability of the true class
    :param r: data row, probability + label
    :return: rank
    """
    l = r[-1]  # label
    probas = r[:-1]
    indices = np.argsort(probas)[::-1]  # larger probability first
    rank = np.where(indices == l)[0]
    return rank[0]


def get_ranks_using_closest_rep(X_train, X_test, y_test):
    n_class = 100
    def predict_vector_rank(vec):
        dis = [closest_class_rep(vec[0], X_train[i::n_class]) for i in range(n_class)]
        # print dis
        dis = np.max(dis) - dis
        a = np.concatenate([dis, [vec[1]]])
        rank = find_rank(a)
        return rank

    # ranks = [predict_vector_rank(x) for x in zip(X_test, y_test)]
    ranks = Parallel(n_jobs=8)(delayed(predict_vector_rank)(x) for x in zip(X_test, y_test))
    return np.asarray(ranks)


def closest_rep_ranking(data_dir, grid_size=16, case=0):

    # case=0: viewing vs. viewing
    # case=1: recall vs. recall
    # case=2: recall vs. others viewing

    X, y = generate_hist_rep(data_dir, size=grid_size, is_viewing=True)
    rX, ry = generate_hist_rep(data_dir, size=grid_size, is_viewing=False)

    ranks = []
    kf = KFold(n_splits=28)
    for train, test in kf.split(X):
        # y_train = y[train]
        if case == 0:
            X_train = X[train]
            X_test = X[test]
            y_test = y[test]
        elif case == 1:
            X_train = rX[train]
            X_test = rX[test]
            y_test = ry[test]
        elif case == 2:
            X_train = X[train]
            X_test = rX[test]
            y_test = ry[test]

        r = get_ranks_using_closest_rep(X_train, X_test, y_test)
        print('mean rank: ', np.mean(r))
        if not len(ranks):
            ranks = r
        else:
            ranks = np.hstack((ranks, r))

    ranks = np.asarray(ranks)
    return ranks.T


def main():

    n = 16  # grid size of histogram
    data_dir = "../../data/raw_samples"

    # case=0: viewing vs. viewing
    # case=1: recall vs. encoding
    # case=2: recall vs. others viewing
    global testing_case
    testing_case = 2

    ranks = closest_rep_ranking(data_dir, grid_size=n, case=testing_case)
    # np.save('case%d.npy'%testing_case, ranks)
    # print(ranks)

    # ranks = np.load('case%d.npy'%testing_case)
    ranks = ranks.reshape(28, -1)
    plot_ranks(ranks, output_name="case%d.pdf"%testing_case)


if __name__ == '__main__':

    main()
