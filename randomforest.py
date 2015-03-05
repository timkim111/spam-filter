import random
import math

FEATURE_SAMPLE_SIZE = 8
STOP_DEPTH = 20

TEST = 1 #prints once per random_forest call

testFeatures = open('testFeatures.csv', 'r')
trainFeatures = open('trainFeatures.csv', 'r')
trainLabels = open('trainLabels.csv', 'r')
valFeatures = open('valFeatures.csv', 'r')
valLabels = open('valLabels.csv', 'r')

def list_str_to_float(lst):
    res = []
    for i in lst:
        res.append(float(i))
    return res

class Dec_Tree:
    # i is index of feature, t is threshold for tnternal nodes, y is class of a leaf
    def __init__(self, i=None, t=None, y=None, left=None, right=None):
        self.i = i
        self.t = t
        self.y = y
        self.left = left
        self.right = right
    def is_empty_tree(self):
        return self.i==None and self.t==None and self.y==None and self.left==None and self.right==None
    def is_leaf(self):
        return self.left == None and self.right == None and self.y != None
    def is_internal_node(self):
        return not self.is_leaf()
    def classify(self, x):
        if self.is_leaf():
            return self.y
        else:
            if x.f[self.i] <= self.t:
                return self.left.classify(x)
            else:
                return self.right.classify(x)

def random_forest(F_L, T, write_file, validation):
    S = t_F_L
    def bagging():
        random_subset = set()
        for i in range(len(S)):
            random_index = random.randint(0, len(S) - 1)
            random_observation = S[random_index]
            random_subset.add(random_observation)
        return list(random_subset)
    def build_dec_tree(S, depth):
        def most_frequent_class():
            count_0 = 0
            count_1 = 0
            for v in S:
                if v.l == 0:
                    count_0 += 1
                elif v.l == 1:
                    count_1 += 1
            if count_1 > count_0: #tie resolved here
                return 1
            else:
                return 0
        def stop():
            if depth >= STOP_DEPTH:
                return True
            if len(S) == 0:
                return True
            class_y = S[0].l
            for i in range(1, len(S)):
                if S[i].l != class_y:
                    return False
            return True
        def H(Y): #Y is a list contain class labels, either 0 or 1
            res = 0.0
            count_0 = 0.0
            count_1 = 0.0
            for y in Y:
                if y == 0:
                    count_0 += 1
                elif y == 1:
                    count_1 += 1
            if count_0 != 0:
                prob_0 = (float(count_0) / (count_0 + count_1))
                res += prob_0 * math.log(prob_0, 2)
            if count_1 != 0:
                prob_1 = (float(count_1) / (count_0 + count_1))
                res += prob_1 * math.log(prob_1, 2)
            return -res
        def goodness(t, feature):
            Y_L = []
            Y_R = []
            Y = []
            for s in S:
                Y.append(s.l)
                if s.f[feature] <= t:
                    Y_L.append(s.l)
                else:
                    Y_R.append(s.l)
            return H(Y) - ((len(Y_L) / len(Y) * H(Y_L)) + (len(Y_R) / len(Y) * H(Y_R)))
        if stop():
            return Dec_Tree(None, None, most_frequent_class(), None, None)
        else:
            features = random.sample(range(len(F_L[0].f)), FEATURE_SAMPLE_SIZE)
            best_feature = 0
            best_t = 0
            best_igm = 0
            for feature in features:
                feature_values = set()
                for s in S:
                    feature_values.add(s.f[feature])
                feature_values = list(feature_values)
                feature_values.sort()
                if len(feature_values) <= 0:
                    break
                elif len(feature_values) == 1:
                    t = feature_values[0]
                    igm = goodness(t, feature)
                    if igm > best_igm:
                        best_feature = feature
                        best_t = t
                        best_igm = igm
                else:
                    for x in range(len(feature_values) - 1):
                        t = (feature_values[x] + feature_values[x+1]) / 2
                        igm = goodness(t, feature)
                        if igm > best_igm:
                            best_feature = feature
                            best_t = t
                            best_igm = igm
            S_L, S_R = [], []
            for s in S:
                if s.f[best_feature] <= best_t:
                    S_L.append(s)
                else:
                    S_R.append(s)
            T_L = build_dec_tree(S_L, depth + 1)
            T_R = build_dec_tree(S_R, depth + 1)
            return Dec_Tree(best_feature, best_t, None, T_L, T_R)

    dec_trees = []
    for i in range(T):
        bagged_S = bagging()
        dec_trees.append(build_dec_tree(bagged_S, 0))
    write_to = open(write_file, 'w')
    hits = 0
    progress_index = 0
    for pair in F_L:
        num_0_votes = 0
        num_1_votes = 0
        for dec_tree in dec_trees:
            found_vote = dec_tree.classify(pair)
            if found_vote == 0:
                num_0_votes += 1
            elif found_vote == 1:
                num_1_votes += 1
        if num_1_votes > num_0_votes: #tie is handled here
            best_choice = 1
        else:
            best_choice = 0
        write_to.write(str(best_choice) + '\n')
        if validation and (best_choice == F_L[progress_index].l):
            hits += 1
        progress_index += 1
    write_to.close()
    if validation:
        if TEST:
            print("T = " + str(T) + ": " + str(float(hits) / len(F_L)))
    
tf = trainFeatures.readlines()
tf = [list_str_to_float(i.strip().split(',')) for i in tf]
tl = trainLabels.readlines()
tl = [int(i.strip()) for i in tl]

vf = valFeatures.readlines()
vf = [list_str_to_float(i.strip().split(',')) for i in vf]
vl = valLabels.readlines()
vl = [int(i.strip()) for i in vl]

testf = testFeatures.readlines()
testf = [list_str_to_float(i.strip().split(',')) for i in testf]
testl = [None for i in range(len(testf))]

#feature_value pair
class F_L_Pair:
    def __init__(self, f, l):
        self.f = f
        self.l = l
#   def __repr__(self):
#       return "P(" + repr(self.f) + ", " + repr(self.l) + ")"

t_F_L = []
for i in range(len(tf)):
    t_F_L.append(F_L_Pair(tf[i], tl[i]))
del tf
del tl

v_F_L = []
for i in range(len(vf)):
    v_F_L.append(F_L_Pair(vf[i], vl[i]))
del vf
del vl

test_F_L = []
for i in range(len(testf)):
    test_F_L.append(F_L_Pair(testf[i], testl[i]))
del testf
del testl

def v_file_name(T):
    return "emailOutput" + str(T) + ".csv"

def test_file_name():
    return "emailOutput.csv"

Tv1 = random_forest(v_F_L, 1, v_file_name(1), True)
Tv2 = random_forest(v_F_L, 2, v_file_name(2), True)
Tv5 = random_forest(v_F_L, 5, v_file_name(5), True)
Tv10 = random_forest(v_F_L, 10, v_file_name(10), True)
Tv25 = random_forest(v_F_L, 25, v_file_name(25), True)

best_T = 25
test_the_set = random_forest(test_F_L, best_T, test_file_name(), False)

testFeatures.close()
trainFeatures.close()
trainLabels.close()
valFeatures.close()
valLabels.close()
