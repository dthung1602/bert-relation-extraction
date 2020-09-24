from matplotlib import pyplot as plt
import json
import numpy as np


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    paths = [
        "data/processed/gids_train.json",
        "data/processed/semeval2010task8_train.json",
    ]

    for i, path in enumerate(paths):
        f = open(path)
        lengths = []
        for line in f.readlines():
            data = json.loads(line)
            lengths.append(sum(data['attention_mask']))

        lengths = np.array(lengths)
        mean = lengths.mean()
        std = np.std(lengths)
        print(mean)

        ax[i].hist(lengths, bins=50)
        ylim = ax[i].get_ylim()
        ax[i].plot([mean, mean], ylim, linestyle='-', label='mean')
        ax[i].plot([mean - std, mean - std], ylim, linestyle='--', label='one standard deviation from mean', color='black')
        ax[i].plot([mean + std, mean + std], ylim, linestyle='--', color='black')
        ax[i].legend()

    fig.show()
