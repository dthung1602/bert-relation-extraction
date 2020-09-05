lines = open("score.txt").readlines()
acc = lines[:4]
macro = lines[4:8]


def avg(x):
    x = [float(t.strip().split("\t")[1]) for t in x]
    return round(100 * sum(x) / len(x), 2)


if __name__ == '__main__':
    print(avg(acc))
    print(avg(macro))
