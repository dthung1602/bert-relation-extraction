import json
import matplotlib.pyplot as plt
from collections import Counter

label_mapping = {
    "/people/deceased_person/place_of_death": "Place of death",
    "/people/person/place_of_birth": "Place of birth",
    "/people/person/education./education/education/institution": "Person-Institution",
    "/people/person/education./education/education/degree": "Person-Degree"
}


def visualize_distribution(dataset_name, figsize):
    with open("data/processed/metadata.json") as f:
        metadata = json.load(f)[dataset_name]

    counters = []
    for subset in ['train', 'val', 'test']:
        counter = Counter()
        with open(f"data/processed/{dataset_name.lower()}_{subset}.json") as f:
            for line in f:
                data = json.loads(line)
                label = metadata['id_to_label'][str(data['label'])]
                if label in label_mapping:
                    label = label_mapping[label]
                counter[label] += 1
        counters.append(counter)

    counters.insert(0, sum(counters, Counter()))
    subsets = ['whole', 'train', 'val', 'test']

    figs = []
    for i, counter, subset in zip([0, 1, 2, 3], counters, subsets):
        no_rel_label = metadata['no_relation_label']
        no_rel_value = counter[no_rel_label]
        labels = list(counter.keys())
        values = list(counter.values())
        no_rel_idx = labels.index(no_rel_label)
        del labels[no_rel_idx]
        del values[no_rel_idx]
        sorted_idx = [i[0] for i in sorted(enumerate(labels), key=lambda x:x[1])]
        sorted_labels = [no_rel_label] + [labels[i] for i in sorted_idx]
        sorted_values = [no_rel_value] + [values[i] for i in sorted_idx]
        total = sum(sorted_values)
        sorted_percent = [value * 100.0 / total for value in sorted_values]

        fig, ax = plt.subplots(figsize=figsize)
        fig.autofmt_xdate(rotation=45)
        figs.append(fig)
        # ax.title.set_text(f"{dataset_name} {subset} dataset label distribution")
        ax.bar(sorted_labels, sorted_values)
        ax.set_ylabel("sentences")
        ax_right = ax.twinx()
        ax_right.bar(sorted_labels, sorted_percent)
        ax_right.set_ylabel("percent")

    return figs


if __name__ == '__main__':
    # figs = visualize_distribution("SemEval2010Task8", figsize=(10, 10))
    figs = visualize_distribution("GIDS", figsize=(6, 6))
    for fig in figs:
        fig.show()
