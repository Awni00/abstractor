import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from scipy.special import binom


class SetGame():

    def __init__(self, verbose=0):
        im = mpimg.imread('./all-cards.png')
        self.cards = im.transpose((1,0,2))
        if (verbose):
            plt.figure(figsize=(10,10))
            plt.imshow(self.cards)
            _ = plt.axis('off')

        (self.leftmargin, self.topmargin) = (4,8)
        (self.vspace, self.hspace) = (1,0)
        (self.height, self.width) = (70, 50)

        self.color = ['red', 'green', 'purple']
        self.pattern = ['empty', 'striped', 'solid']
        self.shape = ['diamond', 'oval', 'squiggle']
        self.number = ['one', 'two', 'three']

        attrs = ['(0,0,0)', '(0,0,1)', '(0,1,0)', '(1,0,0)', '(1,1,1)']
        partial_attrs = [
            '(0,0,0)',
            '(0,0,1)',
            '(0,1,0)',
            '(1,0,0)',
            '(1,1,1)',
            '(*,*,*)',
            '(*,*,0)',
            '(*,*,1)',
            '(*,0,*)',
            '(*,0,0)',
            '(*,0,1)',
            '(*,1,*)',
            '(*,1,0)',
            '(*,1,1)',
            '(0,*,*)',
            '(0,*,0)',
            '(0,*,1)',
            '(0,0,*)',
            '(0,1,*)',
            '(1,*,*)',
            '(1,*,0)',
            '(1,*,1)',
            '(1,0,*)',
            '(1,1,*)']
        self.voc = {attrs[j]:j for j in range(len(attrs))}
        self.ivoc = {j:attrs[j] for j in range(len(attrs))}
        self.partial_voc = {partial_attrs[j]:j for j in range(len(partial_attrs))}
        self.partial_ivoc = {j:partial_attrs[j] for j in range(len(partial_attrs))}
        self.matches = {
            '(0,0,0)': set([0]),
            '(0,0,1)': set([1]),
            '(0,1,0)': set([2]),
            '(1,0,0)': set([3]),
            '(1,1,1)': set([4]),
            '(*,*,*)': set([0,1,2,3]),
            '(*,*,0)': set([0,2,3]),
            '(*,*,1)': set([1,4]),
            '(*,0,*)': set([0,1,3]),
            '(*,0,0)': set([0,3]),
            '(*,0,1)': set([1]),
            '(*,1,*)': set([2,4]),
            '(*,1,0)': set([2]),
            '(*,1,1)': set([4]),
            '(0,*,*)': set([0,1,2]),
            '(0,*,0)': set([0,2]),
            '(0,*,1)': set([1]),
            '(0,0,*)': set([0,1]),
            '(0,1,*)': set([2]),
            '(1,*,*)': set([3,4]),
            '(1,*,0)': set([3]),
            '(1,*,1)': set([4]),
            '(1,0,*)': set([3]),
            '(1,1,*)': set([4])}
        self.X, self.y, self.triples = self.generate_grouped_data(verbose)
        if (verbose):   
            self.display_samples()

        self.num_actions = 3
        self.state_dimension = 4
        self.state_values = len(self.partial_voc)

    class State:

        def __init__(self, env, dealt_cards, rewards=None, verbose=0):
            self.num_cards = len(dealt_cards)

            self.focus_indices = list()
            for i in np.arange(self.num_cards):
                for j in np.arange(i+1, self.num_cards):
                    for k in np.arange(self.num_cards):
                        if (k==i) or (k==j) :
                            continue
                        self.focus_indices.append([i,j,k])

            self.current_focus = 0
            self.focus_ind = self.focus_indices[self.current_focus]
            focus_ind = self.focus_ind
            focus_triple = [dealt_cards[focus_ind[0]], 
                            dealt_cards[focus_ind[1]],
                            dealt_cards[focus_ind[2]]]
            attr = env.tabulate_attributes_for_triple(focus_triple)

            self.dealt_cards = dealt_cards
            self.focus_ind = focus_ind
            self.focus_triple = focus_triple
            self.focus_attributes = attr
            self.partial_attributes = len(attr)*[-1]
            self.attribute_ind = 0

            self.set_reward = 3 * int(binom(self.num_cards, 3))
            self.notset_reward = -24
            self.default_reward = -1

        def advance(self, env, action=0):

            if action == 0 :  # measure attribute
                self.partial_attributes[self.attribute_ind] = self.focus_attributes[self.attribute_ind]
                self.attribute_ind = (self.attribute_ind + 1) % 12
                return (env.default_reward, False)

            elif action == 1 : # swap in a new card
                prev_ind = self.focus_ind
                self.current_focus = (self.current_focus + 1) % len(self.focus_indices)
                focus_ind = self.focus_indices[self.current_focus]
                focus_triple = [self.dealt_cards[focus_ind[0]], 
                                self.dealt_cards[focus_ind[1]],
                                self.dealt_cards[focus_ind[2]]]
                attr = env.tabulate_attributes_for_triple(focus_triple)

                self.focus_ind = focus_ind
                self.focus_triple = focus_triple
                self.focus_attributes = attr

                partial_attr = self.partial_attributes
                if not(prev_ind[0] == focus_ind[0]):
                    for j in [0, 3, 6, 9]:
                        partial_attr[j] = -1
                        partial_attr[j+1] = -1
                if not(prev_ind[1] == focus_ind[1]):
                    for j in [0, 3, 6, 9]:
                        partial_attr[j] = -1
                        partial_attr[j+2] = -1
                if not(prev_ind[2] == focus_ind[2]) :
                    for j in [0, 3, 6, 9]:
                        partial_attr[j+1] = -1
                        partial_attr[j+2] = -1
                self.partial_attributes = partial_attr
                return (env.swap_reward, False)

            elif action == 2 : # declare SET!
                if env.triple_is_set(self.focus_triple):
                    return (env.set_reward, True)
                else:
                    return (env.notset_reward, False)


    def set_rewards(self, rewards, verbose=0):
        self.set_reward = rewards[0]
        self.notset_reward = rewards[1]
        self.swap_reward = rewards[2]
        self.default_reward = rewards[3]
        if verbose :
            print("Rewards: set=%d, notset=%d, swap=%d, default=%d" % (self.set_reward, self.notset_reward, self.swap_reward, self.default_reward))

    def init_state(self, num_cards=12, verbose=0, shuffle=True):
        inds = (self.y==True)
        posi = np.arange(len(self.y))[inds]
        set_index = np.random.choice(posi, size=1)[0]

        card_coord = [(i,j) for i in np.arange(9) for j in np.arange(9)]

        while(True):
            inds = np.random.choice(np.arange(81), size=num_cards-3, replace=False)
            if len(set(self.triples[set_index]) & set([card_coord[j] for j in inds])) == 0:
                break

        dealt_cards = self.triples[set_index] + [card_coord[j] for j in inds]
        if shuffle:
            random.shuffle(dealt_cards)
        self.state = self.State(self, dealt_cards, verbose=verbose)

        partial_attr = self.str_encode_attributes(self.state.partial_attributes)
        obs = np.array([self.partial_voc[partial_attr[j]] for j in range(4)])
        return obs, 0, False

    def advance_state(self, action):
        reward, done = self.state.advance(self, action)
        partial_attr = self.str_encode_attributes(self.state.partial_attributes)
        obs = np.array([self.partial_voc[partial_attr[j]] for j in range(4)])
        return obs, reward, done

    def show_state(self):
        labels = len(self.state.dealt_cards)*['']
        labels[self.state.focus_ind[0]] = 'A'
        labels[self.state.focus_ind[1]] = 'B'
        labels[self.state.focus_ind[2]] = 'C'

        self.show_cards(self.state.dealt_cards, 3, int(np.ceil(self.state.num_cards/3)), labels)
        self.show_triple(self.state.focus_triple)

        attr = self.str_encode_attributes(self.state.focus_attributes)
        print(attr)
        partial_attr = self.str_encode_attributes(self.state.partial_attributes)
        print(partial_attr)

    def image_of_card(self, row, col):
        topleft = np.array([self.leftmargin + row*(self.height + self.hspace), \
            self.topmargin + col*(self.width + self.vspace)])
        bottomright = topleft + [self.height, self.width]
        return self.cards[topleft[0]:bottomright[0], topleft[1]:bottomright[1], :]

    def show_card(self, row, col):
        c = self.image_of_card(row, col)
        plt.figure(figsize=(2,2))
        plt.imshow(c)
        _ = plt.axis('off')
        plt.show()

    def attributes_of_card(self, row, col):
        return (self.number[row % 3], self.color[int(col/3)], self.pattern[col % 3], self.shape[int(row/3)])

    def show_cards(self, cards, nrow, ncol, labels=[]):
        fig, axarr = plt.subplots(nrow, ncol, figsize=(1.3*ncol, 1.3*nrow))
        for i in np.arange(len(cards)):
            imcol = i % ncol
            imrow = int(i/ncol) % nrow
            (row, col) = (cards[i][0], cards[i][1])
            c = self.image_of_card(row, col)
            axarr[imrow, imcol].imshow(c)
            axarr[imrow, imcol].axis('off')
            if len(labels) > 0:
                axarr[imrow, imcol].set_title(labels[i])
        for i in np.arange(len(cards), nrow*ncol):
            imcol = i % ncol
            imrow = int(i/ncol) % nrow
            axarr[imrow, imcol].axis('off')
        fig.tight_layout()
        plt.show()

    def show_triple(self, cards):
        fig, axarr = plt.subplots(1, 3, figsize=(3, 2))
        label = ['A', 'B', 'C']
        for i in np.arange(len(cards)):
            (row, col) = (cards[i][0], cards[i][1])
            c = self.image_of_card(row, col)
            axarr[i].imshow(c)
            axarr[i].axis('off')
            axarr[i].set_title('%s' % label[i])
        fig.tight_layout()
        plt.show()

    def tabulate_features_for_pair(self, A, B):
        A_attr = np.array(self.attributes_of_card(A[0], A[1]))
        B_attr = np.array(self.attributes_of_card(B[0], B[1]))
        return A_attr == B_attr

    def display_samples(self):

        # a single card
        self.show_card(1,5)

        # attributes of cards
        card_coord = [(i,j) for i in np.arange(9) for j in np.arange(9)]
        for i in np.random.choice(np.arange(81), size=3):
            (row, col) = card_coord[i]
            self.show_card(row, col)
            print(self.attributes_of_card(row, col))

        # board of cards
        inds = np.random.choice(np.arange(81), size=12, replace=False)
        dealt_cards = [card_coord[j] for j in inds]
        self.show_cards(dealt_cards, 3, 4)
        triple = [dealt_cards[0], dealt_cards[1], dealt_cards[2]]
        self.show_triple(triple)
        a = self.tabulate_attributes_for_triple(triple)
        self.display_attributes(a)

    
    def tabulate_features_for_triple(self, triple):
        AB = self.tabulate_features_for_pair(triple[0], triple[1])
        AC = self.tabulate_features_for_pair(triple[0], triple[2])
        BC = self.tabulate_features_for_pair(triple[1], triple[2])
        return list(AB) + list(AC) + list(BC)

    def tabulate_attributes_for_triple(self, triple):
        AB = self.tabulate_features_for_pair(triple[0], triple[1])
        AC = self.tabulate_features_for_pair(triple[0], triple[2])
        BC = self.tabulate_features_for_pair(triple[1], triple[2])
        attr = list()
        for j in np.arange(4):
            attr = attr + [AB[j], AC[j], BC[j]]
        return attr

    def attribute_is_good(self, a, b, c):
        return (a and b and c) | (not(a) and not(b) and not(c))

    def triple_is_set(self, triple):
        AB = self.tabulate_features_for_pair(triple[0], triple[1])
        AC = self.tabulate_features_for_pair(triple[0], triple[2])
        BC = self.tabulate_features_for_pair(triple[1], triple[2])
        is_set = self.attribute_is_good(AB[0], AC[0], BC[0]) and \
                 self.attribute_is_good(AB[1], AC[1], BC[1]) and \
                 self.attribute_is_good(AB[2], AC[2], BC[2]) and \
                 self.attribute_is_good(AB[3], AC[3], BC[3])
        return is_set

    def display_features(self, v):
        vals = dict()
        offset = 0
        pair = ['A,B', 'A,C', 'B,C']
        attribute = ['number', 'color', 'pattern', 'shape']
        for i in np.arange(3):
            for j in np.arange(4):
                vals['%s(%s)' % (attribute[j], pair[i])] = v[4*i+j] 
        print(vals)

    def display_attributes(self, v):
        vals = dict()
        offset = 0
        pair = ['A,B', 'A,C', 'B,C']
        attribute = ['number', 'color', 'pattern', 'shape']
        for j in np.arange(4):
            for i in np.arange(3):
                vals['%s(%s)' % (attribute[j], pair[i])] = v[3*j+i] 
        print(vals)

    def str_encode_attributes(self, a):
        words = list()
        for j in np.arange(4):
            vals = '(%d,%d,%d)' % (a[3*j+0], a[3*j+1], a[3*j+2])
            vals = vals.replace('-1','*')
            words = words + [vals]
        return words

    def generate_data(self):
        n = 85320 # 81 choose 3
        X = np.array((n*12)*[0]).reshape(n, 12)
        y = np.array(n*[0])
        sets = 0
        non_sets = 0
        triples = list()
        t = 0

        for i in np.arange(81):
            c1 = (int(i/9), i%9)
            for j in np.arange(i+1, 81):
                c2 = (int(j/9), j%9)
                for k in np.arange(j+1, 81):
                    c3 = (int(k/9), k%9)
                    triple = [c1, c2, c3]
                    np.random.shuffle(triple)
                    this_x = self.tabulate_attributes_for_triple(triple)
                    this_y = self.triple_is_set(triple)
                    X[t,:] = this_x
                    y[t] = this_y
                    triples.append(triple)
                    t = t + 1
        print('Total number of triples: %d' % t)
        print('Probability of SET! (in %d samples): %f (1/79=%f)' % (n, sum(y) / len(y), 1/79))
        return X, y, triples

    def generate_grouped_data(self, verbose=0):
        n = 85320 # 81 choose 3
        triples = list()
        X = np.array((n*4)*[0]).reshape(n, 4)
        y = np.array(n*[0])
        sets = 0
        non_sets = 0
        triples = list()
        t = 0

        for i in np.arange(81):
            c1 = (int(i/9), i%9)
            for j in np.arange(i+1, 81):
                c2 = (int(j/9), j%9)
                for k in np.arange(j+1, 81):
                    c3 = (int(k/9), k%9)
                    triple = [c1, c2, c3]
                    np.random.shuffle(triple)
                    atts = self.tabulate_attributes_for_triple(triple)
                    words = self.str_encode_attributes(atts)
                    this_x = [self.voc[words[j]] for j in np.arange(4)]
                    this_y = self.triple_is_set(triple)
                    X[t,:] = this_x
                    y[t] = this_y
                    t = t + 1
                    triples.append(triple)
        if verbose :
            print('Total number of triples: %d' % t)
            print('Probability of SET! (in %d samples): %f (1/79=%f)' % (n, sum(y) / len(y), 1/79))

        return X, y, triples

    def data_matches_attributes(self, x, a):
        for j in np.arange(4):
            if not(x[j] in self.matches[a[j]]):
                return False
        return True

    def str_encode_data_point(x):
        s = list()
        for j in np.arange(4):
            s.append(self.ivoc[x[j]])
        return s

