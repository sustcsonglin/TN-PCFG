from nltk import Tree
import argparse
import pickle


def factorize(tree):
    def track(tree, i):
        label = tree.label()
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            return (i+1 if label is not None else i), []
        j, spans = i, []
        for child in tree:
            j, s = track(child, j)
            spans += s
        if label is not None and j > i:
            spans = [[i, j, label]] + spans
        elif j > i:
            spans = [[i, j, 'NULL']] + spans
        return j, spans
    return track(tree, 0)[1]


def create_dataset(file_name):
    word_array = []
    pos_array = []
    gold_trees = []
    with open(file_name, 'r') as f:
        for line in f:
            tree = Tree.fromstring(line)
            token = tree.pos()
            word, pos = zip(*token)
            word_array.append(word)
            pos_array.append(pos)
            gold_trees.append(factorize(tree))

    return {'word': word_array,
            'pos': pos_array,
            'gold_tree':gold_trees}





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='preprocess ptb file.'
    )
    parser.add_argument('--train_file', default='data/ptb-train.txt')
    parser.add_argument('--val_file', default='data/ptb-valid.txt')
    parser.add_argument('--test_file', default='data/ptb-test.txt')
    parser.add_argument('--cache_path', default='data/')

    args = parser.parse_args()

    result = create_dataset(args.train_file)
    with open(args.cache_path+"train.pickle", "wb") as f:
        pickle.dump(result, f)

    result = create_dataset(args.val_file)
    with open(args.cache_path+"val.pickle", "wb") as f:
        pickle.dump(result, f)

    result = create_dataset(args.test_file)
    with open(args.cache_path+"test.pickle", "wb") as f:
        pickle.dump(result, f)











