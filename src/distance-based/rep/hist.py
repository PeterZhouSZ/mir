import csv
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import os


screen_resolution = [1920, 1200]


def generate_heatmap_from_seq(seq, grid_size=12, norm_type=1, to_plot=False):
    """ Generate heatmap from sequence of raw samples
    :param seq: sequence of raw samples, each row [timestamp, x, y]
    :param grid_size: number of bins in histogram
    :param norm_type: type of normalization
    :param to_plot: toggle plotting
    :return:
    """

    x = seq[:, 1]
    y = seq[:, 2]
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[grid_size, grid_size],
                                             range=[[0, screen_resolution[0]],
                                                    [0, screen_resolution[1]]])
    if norm_type == 1:
        heatmap /= np.sum(heatmap)
    elif norm_type == 2:
        heatmap = np.square(heatmap).astype(np.float32)
        heatmap /= np.sum(heatmap)

    if to_plot:
        print (heatmap.T)
        fig, arr = plt.subplots(1, 2)
        arr[0].scatter(seq[:, 1], seq[:, 2])
        arr[0].set_xlim(0, 1920)
        arr[0].set_ylim(1200, 0)
        arr[0].set_aspect('equal')
        arr[1].matshow(heatmap.T)
        plt.show()

    return heatmap.T  # origin at left bottom corner


def cvs_to_dict(path):
    data_dict = {}
    idx = -1

    with open(path, newline='') as f:
        rows = csv.reader(f, delimiter=',')
        for r in rows:
            if "Image" in r[0]:
                if idx > -1:
                    data_dict[idx] = np.asarray(data_dict[idx])
                idx += 1
                data_dict[idx] = []
            else:
                t = float(r[0])
                x = float(r[1])
                y = float(r[2])
                data_dict[idx].append([t, x, y])

    data_dict[idx] = np.asarray(data_dict[idx])
    # print(data_dict.keys())
    # print(data_dict[0])
    return data_dict


def generate_hist_rep(data_dir, is_viewing=True, size=12):
    """ Generate histogram representation of the whole data set
    :param is_viewing: viewing or recall seqences
    :param size: grid size of histogram
    :return: hist_mat and labels
        hist_mat 2800 * (size*size), each row stores the flattened vector of heat map.
                0-99 observer 1 100-199 observer 2 etc.
        labels 2800 corresponding labels of class, stimulus index
    """

    hist_dim = size * size

    # import time
    # st = time.time()
    def process_single_data(idx):
        hms = np.zeros([100, hist_dim])
        if is_viewing:
            raw_sample_file_name = "v%d.csv" % idx
        else:
            raw_sample_file_name = "r%d.csv" % idx

        path = os.path.join(data_dir, raw_sample_file_name)
        data_dict = cvs_to_dict(path)

        for i in range(100):
            hm = generate_heatmap_from_seq(data_dict[i], grid_size=size)
            hms[i, :] = hm.flatten()
        return hms

    hms = Parallel(n_jobs=8)(delayed(process_single_data)(x) for x in range(28))
    # print(time.time()-st)
    # # print(hms)
    hist_mat = np.vstack(hms)

    l = np.arange(100).reshape(1, 100)
    labels = np.repeat(l, 28, axis=0)
    labels = labels.flatten()

    # hist_mat = np.zeros([100*28, hist_dim])
    # labels = np.zeros(100*28)
    # st = time.time()
    # for idx in range(28):
    #     if is_viewing:
    #         raw_sample_file_name = "v%d.csv" % idx
    #     else:
    #         raw_sample_file_name = "r%d.csv" % idx
    #
    #     path = os.path.join(data_dir, raw_sample_file_name)
    #     data_dict = cvs_to_dict(path)
    #     for i in range(100):
    #         ridx = idx * 100 + i
    #         hm = generate_heatmap_from_seq(data_dict[i], grid_size=size)
    #         hist_mat[ridx, :] = hm.flatten()
    #         labels[ridx] = i
    # print(time.time()-st)

    return hist_mat, labels


if __name__ == '__main__':
    p = "../../../data/raw_samples/v0.csv"
    cvs_to_dict(p)

    data_dir = "../../../data/raw_samples/"
    r = generate_hist_rep(data_dir, size=3)
    print(r[0].shape)
    print(r[1].shape)