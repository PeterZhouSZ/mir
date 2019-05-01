import collections
import matplotlib.pyplot as plt
import numpy as np
import os


def get_cumulative_sum(r):
    """ Calculate the cumulative sum
    :param r: vector of ranks
    :return: vector of cumulative sum
    """
    ranks = np.asarray(r)
    counter = collections.Counter(ranks)
    curve = np.zeros(len(ranks))
    indices = np.fromiter(counter.keys(), dtype=int)
    curve[indices] = np.fromiter(counter.values(), dtype=float)
    curve = np.cumsum(curve)
    return curve


def plot_ranks(ranks, output_name='test.pdf'):
    all_data = []
    for row in ranks:
        all_data.append(get_cumulative_sum(row))
    all_data = np.asarray(all_data)
    mean_curve = np.median(all_data, axis=0)

    sort_res = np.sort(all_data, axis=0)
    mid_idx = int(0.25 * ranks.shape[0])

    low_y = sort_res[mid_idx]
    upper_y = sort_res[-(mid_idx + 1)]

    low_y = np.insert(low_y, 0, 0)
    upper_y = np.insert(upper_y, 0, 0)
    curve = np.insert(mean_curve, 0, 0)
    fig, ax = plt.subplots(figsize=(8,6))
    indices = np.arange(0, 100+1)

    # perform the Mann-Whitney rank test to compute the significance of the observed roc curve
    import scipy.stats as stats
    print (stats.mannwhitneyu(curve, indices, use_continuity=True, alternative='two-sided'))

    ax.plot(indices, indices, '-', linewidth=2, color='0.5')
    ax.fill_between(indices, low_y, upper_y, color='#b9cfe7', edgecolor='')
    ax.plot(indices, upper_y, '-', color='#b9cfe7')

    ax.plot(indices, curve, linewidth=2, color='0.1')
    dis1 = (np.sum(curve) - 0.5 * curve[-1]) / 100.0

    auc = 'AUC = %.1f%%' % dis1
    ax.text(60, 55, auc)  # report area under curve as text label

    ax.set_xlim([0, 100])
    ax.set_ylim([0, 105])
    # Borders
    ax.spines['top'].set_color('0.5')
    ax.spines['bottom'].set_color('0.5')
    ax.spines['left'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.get_yaxis().set_tick_params(direction='out')
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.grid(linestyle='dashed')
    ax.legend(loc='lower right')

    # import matplotlib
    # matplotlib.rcParams.update({'font.size': 16})
    # plt.title(title_str)
    plt.xlabel('Ranks')
    plt.ylabel('Number of instances')

    # legend = plt.legend(loc=9, bbox_to_anchor=(0, -0.21, 1., .102), ncol=3, mode='expand')
    # frame = legend.get_frame().set_edgecolor('0.5')
    plt.ylim(0, 100 + 1)
    plt.tight_layout()

    data_dir = "."
    image_path = os.path.join(data_dir, output_name)
    # plt.savefig(image_path, bbox_extra_artists=(legend,), bbox_inches='tight', dpi=150)
    plt.savefig(image_path, bbox_inches='tight', dpi=150)
    print ("save figure: ", image_path)

    plt.show()