import numpy as np
import random

class Tree:
    def __init__(self, min_leaf_size=1, max_depth=5):
        self.root = None
        self.words = None
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth

    def fit(self, words, verbose=False):
        self.words = words
        self.root = Node(depth=0, parent=None)

        self.root.fit(all_words=self.words, my_words_idx=np.arange(len(self.words)),
                      min_leaf_size=self.min_leaf_size, max_depth=self.max_depth, verbose=verbose)

    def predict(self, bigrams, max_words=5):
        return self.root.predict(bigrams, max_words)


class Node:
    def __init__(self, depth, parent):
        self.depth = depth
        self.parent = parent
        self.all_words = None
        self.my_words_idx = None
        self.children = {}
        self.is_leaf = True
        self.query = None
        self.history = []

    def get_query(self):
        return self.query

    def get_child(self, response):
        if self.is_leaf:
            return self
        else:
            if response not in self.children:
                response = list(self.children.keys())[0]
            return self.children[response]

    def get_bigrams(self, word, lim=5):
        bg = [''.join(bg) for bg in zip(word, word[1:])]
        bg = sorted(set(bg))
        return tuple(bg)[:lim]

    def get_random_bigram(self):
        return chr(ord('a') + random.randint(0, 25)) + chr(ord('a') + random.randint(0, 25))

    def process_leaf(self, all_words, my_words_idx, history, verbose):
        self.my_words_idx = my_words_idx

    def process_node(self, all_words, my_words_idx, history, verbose):
        query = self.get_random_bigram()
        split_dict = {True: [], False: []}

        for idx in my_words_idx:
            bg_list = self.get_bigrams(all_words[idx])
            split_dict[query in bg_list].append(idx)

        return query, split_dict

    def fit(self, all_words, my_words_idx, min_leaf_size, max_depth, fmt_str="    ", verbose=False):
        self.all_words = all_words
        self.my_words_idx = my_words_idx

        if len(my_words_idx) <= min_leaf_size or self.depth >= max_depth:
            self.is_leaf = True
            self.process_leaf(self.all_words, self.my_words_idx, self.history, verbose)
        else:
            self.is_leaf = False
            self.query, split_dict = self.process_node(self.all_words, self.my_words_idx, self.history, verbose)

            for i, (response, split) in enumerate(split_dict.items()):
                self.children[response] = Node(depth=self.depth + 1, parent=self)
                history = self.history.copy()
                history.append(self.query)
                self.children[response].history = history
                self.children[response].fit(self.all_words, split, min_leaf_size, max_depth, fmt_str, verbose)

    def predict(self, bigrams, max_words=5):
        node = self
        valid_words = []

        def contains_all_bigrams(word, bigrams):
            word_bigrams = self.get_bigrams(word)
            return all(bg in word_bigrams for bg in bigrams)

        while len(valid_words) < max_words and not node.is_leaf:
            node = node.get_child(any(bg in bigrams for bg in node.get_bigrams(self.all_words[node.my_words_idx[0]])))

        for idx in node.my_words_idx:
            word = self.all_words[idx]
            if contains_all_bigrams(word, bigrams):
                valid_words.append(word)
                if len(valid_words) == max_words:
                    break

        return valid_words

################################
# Non Editable Region Starting #
################################
def my_fit(words):
################################
#  Non Editable Region Ending  #
################################
    tree = Tree(min_leaf_size=1, max_depth=5)
    tree.fit(words)
    return tree

################################
# Non Editable Region Starting #
################################
def my_predict(model, bigram_list):
################################
#  Non Editable Region Ending  #
################################
    return model.predict(bigram_list)
