import numpy as np
import matplotlib
matplotlib.use('Agg')
import argparse
import itertools
import sys, time
import matplotlib.pyplot as mp
import matplotlib.colors as mc
from math import sqrt
from operator import itemgetter
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
# from sklearn.lda import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
import os
# This is an experiment object, that handles experimentation on RAP-BIDAL
# data, for the sake of the RAP project, this is the more generalized version
# than the original RAPTEST.py it is derived from, however, it can and will
# perform the same tasks as the original.
#

import warnings
warnings.filterwarnings("ignore")
class Experiment(object):

    # Instantiates an experiment object, by running experiments on
    # some given filepath
    # Requires:
    #   fp - the location of the experimental data, with folds indicated
    #   by the number of 
    # Optional:
    #   classifier - classifier chosen for experimentation
    #
    def __init__(self, fp, classifier = RandomForestClassifier, tune = False, \
            batch = False, search_area = None, tune_loc= None, labels=None):

        self.fp = fp
        self.matrices = {}
        self.cl = classifier

        results = []
        self.tmp_labels = labels
        self.act_labels = None

        print("DEBUG: self.fp: ", self.fp)
        print("DEBUG: self.cl: ", self.cl)
        
        #Run the appropriate tuning
        if tune and tune_loc and search_area: 
            if batch:
                self.params = self.batch_tune(search_area, tune_loc)
            else:
                self.load_data(tune_loc)
                self.params = self.exhaustive_param_tune(search_area)
        else:
            self.params = None

        if batch:
            if self.params is None:
                results_raw = self.batch_test(parameters=None, location=self.fp)
            else:
                results_raw = self.batch_test(self.params[-1][-1], self.fp)
            for a in results_raw:
                results.append(results_raw[a])
                #print([np.argmax(j)for j in sorted(results[-1][-1], key=int)],
                #  results[-1][0])
                self.matrices[a] = confusion_matrix([np.argmax(j)for j in \
                    results[-1][-1]], results[-1][0])
        else:
            self.load_data(self.fp)
            self.cl_title = self.fp[-5:-1] +  time.strftime("%d-%m-%Y-%H%M%S", time.localtime())
            for each in self.data:
                #This is the worst python ever written
                start = time.perf_counter()
                results.append(self.test_fold(each, self.labels) if \
                        self.params is 
                        None else self.test_fold(each, self.labels, \
                                self.params[-2][-1], nandetector=True))
                self.time = time.perf_counter() - start
                self.matrices[each] = confusion_matrix([np.argmax(j)\
                        for j in results[-1][-1]], results[-1][0])
        roc_preds , roc_probs =  [], []
        for each in results:
            metrics_accumulated = []
            for i in range(len(each[0])):
                roc_preds.append([1 if j is int(each[0][i]) else 0 \
                        for j in sorted([int(k) for k in \
                        self.labels], key=int)])
                roc_probs.append(each[-1][i])
        roc_preds = np.array(roc_preds)
        roc_probs = np.array(roc_probs)
        # Ugly bullshit
        for k, y in zip(roc_probs, roc_preds):
            z = ' '.join([str(i) for i in k])
            if 'nan' in z:
                pass
                #print(z + str(y))
        roc_rates = []
        for k in range(roc_preds.shape[1]):
            fpr, tpr, thresh = roc_curve(roc_preds[:,k], roc_probs[:,k])
            roc_rates.append((fpr,tpr,thresh))
        roc_auc = [auc(i[0], i[1]) for i in roc_rates]
        self.roc_avg = sum(roc_auc) /len(roc_auc)
        self.graph_it(roc_rates, roc_auc)
        self.print_results()


    # Checks a given filepath for test data, and prepares the output directory
    # Requires:
    #   fp - the location of the test files, every file in this directory will
    #       be treated as a separate fold, and an output directory will be made
    #       in
    #       the current working directory with the name fp + '-results'
    # Optional:
    #   mk_out - whether or not to create an output directory, defaults to true
    #
    def load_data(self, fp, mk_out=True):
        self.data, classes = {}, {}
        st = ''
        #for each in os.listdir(fp):
        for each in [i for i in os.listdir(fp)]:
            print('reading ' + each)
            self.data[each] = np.genfromtxt(os.path.join(fp,each))
            if self.tmp_labels is not None:
                d = dict()
                d2 = dict()
                counter = 0
                #janky fix for subsets of labels
                for k in self.data[each]:
                    if k[0] not in d2 and k[0] in self.tmp_labels:
                        d2[k[0]] = counter
                        d[d2[k[0]]] = []
                        print(d2[k[0]])
                        counter +=1
                    if k[0] in self.tmp_labels:
                        d[d2[k[0]]].append(k[1:])
                else:
                    a = []
                    for i in d:
                        for k in d[i]:
                            a.append([i]+list(k))
                    self.data[each] = np.array(a)
                    self.act_labels = {}
                    for k in d2:
                        self.act_labels[d2[k]]=k
            st = st if st is not '' else each
            st = each
            print(st)
            st = ''.join([st[i] if st[i] == each[i] else '/' for i in range(len(st))])
        else:
            for i in self.data[list(self.data.keys())[1]][:,0]:
                classes[i] = 1
        self.title = st.replace('/','').replace('.txt','').replace('.','')
        self.labels = classes.keys()
        if mk_out:
            self.expdir = fp.split('/')[-3] + '-results/' if fp.endswith('/') else fp.split('/')[-1]
            try:
                os.mkdir(self.expdir)
            except:
                print('A: Failed to make output directory, assuming it exists already')

    # Performs a test on a given fold with a given set of labels
    # Requires:
    #   fold - label of the fold that should be tested
    #   labels - list of all the possible labels for a given test, may be superfluous now that partial fit
    #
    def test_fold(self, fold, labels, clargs=None, nandetector=False):
        cl = self.cl() if clargs is None else self.cl(**clargs)
        a = [[self.data[i][:,1:32],self.data[i][:,0]] for i in self.data if i is not fold]
        cl.fit(list(itertools.chain.from_iterable([i[0] for i in a])),\
                list(itertools.chain.from_iterable([[int(z) for z in i[1]] for i in a])))
        #ENFUCKULATE
        x = cl.score(self.data[fold][:,1:32], [int(z) for z in self.data[fold][:,0]])
        for k in self.data[fold]:

            a = cl.predict_proba(k[1:32].reshape(1, -1))
            # a = cl.predict_proba(k[1:32])
            if True in np.isnan(a) and nandetector is True:
                print(int(k[33]))
        return [[int(z) for z in self.data[fold][:,0]], x,cl.predict_proba(self.data[fold][:,1:32])]

    # Prints the ROC and AUC graphs to self.expdir/self.title-ROC.png
    # Requires:
    #   rates - list of true positive rates index, and false positive rates for every gesture
    #       First set of indices chooses between gestures, second set of indices are to choose
    #       from the following:
    #           0 : true negative rate for a certain threshold
    #           1 : true positive rate for a certain threshold
    #   auc - list of the areas under the curve, for a given index. This must match with the
    #       list of indices in rates
    #
    # Returns nothing
    def graph_it(self, rates, auc):
        a = mp.figure(figsize=(20,15), dpi=200)
        colors = list(mc.cnames.keys())[:len(rates)]
        for i in range(len(rates)):
            if self.act_labels is not None:
                mp.plot(rates[i][0], rates[i][1], color=colors[i], \
                    lw=2, label='Gesture' + str(self.act_labels[i]) + ' , AUC = ' + str(auc[i]))
            else:
                mp.plot(rates[i][0], rates[i][1], color=colors[i], \
                    lw=2, label='Gesture' + str(i) + ' , AUC = ' + str(auc[i]))
        mp.xlim([0.0, 1.0])
        mp.ylim([0.0, 1.05])
        mp.xlabel('False Positive Rate')
        mp.ylabel('True Positive Rate')
        mp.title(self.cl.__name__ + ' ROC')
        mp.legend(loc="lower right", prop={ 'size':8 })
        a.savefig(self.expdir+ '/' + self.cl_title + '-ROC.png')
        mp.close('all')

    # Exhaustively tunes parameters of classifier
    # Requires:
    #   parameters - Dictionary whose keys are parameter variable names and whose values are
    #    paired with lists of possible values
    # 
    def exhaustive_param_tune(self, parameters, hist_size = 5):
        param_names  = []
        param_values = []
        param_hist = []
        for j in parameters:
            param_names.append(j)
            param_values.append(parameters[j])
        param_values = itertools.product(*param_values)
        for k in param_values:
            param_dict = {}
            for i, j in enumerate(param_names):
                param_dict[j] =  k[i]
            z = 0
            for each in self.data:
                z += self.test_fold(each, self.labels, clargs=param_dict)[1]
            z = z / len(self.data)
            print('recorded z is :' + str(z))
            param_hist.append((z,  param_dict))
            param_hist = sorted(param_hist, key=lambda x : float(x[0]))[-1*hist_size:]
        print(param_hist)
        return param_hist
        #Needs the accuracy from the results section

    # Tunes on batches of data
    # Requires:
    #   parameters - Dictionary whose keys are parameter variable names and whose values are
    #    paired with lists of possible values
    # 
    #   location - Location of datasets to tune on
    #
    def batch_tune(self, parameters, location, store=True):
        param_names  = []
        param_values = []
        data_sets = {}
        param_hist = {}
        for j in parameters:
            param_names.append(j)
            param_values.append(parameters[j])
        param_values = itertools.product(*param_values)
        for k in os.listdir(location):
            self.load_data(location + '/'+k, mk_out=False)
            data_sets[k] = self.data
        start = time.perf_counter()
        for i in param_values:
            t_a = 0
            t_r = 0
            param_dict = {}
            for k, j in enumerate(param_names):
                param_dict[j] =  i[k]
                print(param_dict)
            for k in data_sets:
                self.data = data_sets[k]
                a = 0
                r = 0
                for each in self.data:
                    a += self.test_fold(each, self.labels, clargs=param_dict)[1]
                    r += 1
                t_a += a
                t_r += r
                a = a / float(r)
                if k not in param_hist:
                    param_hist[k] =  []
                param_hist[k].append([a, param_dict])
                print('accuracy : ' + str(a))
                print(param_dict)
            t_a = t_a / t_r
            if location not in param_hist:
                param_hist[location] = []
            param_hist[location].append([t_a, param_dict])
        t = str(time.perf_counter()-start)
        print('Tune time: ' + t)
        if store:
            cl_title = self.fp[-5:-1] + time.strftime("%d-%m-%Y-%H%M%S", time.localtime())
            with open('params_'+ cl_title +'.txt', 'w') as f:
                f.write('Tune time : ' + t + '\n' + '##############################' + '\n')
                for a in param_hist.keys():
                    f.write(a + '\n' + '##############################')
                    param_hist[a] = sorted(param_hist[a], key=lambda x : float(x[0]))
                    for k in param_hist[a]:
                        f.write(str(k[0]) + ',' + ','.join([i + ':' + str(k[1][i]) for i in k[1]])+ '\n')
        hist_size = 5
        param_hist[location] = sorted(param_hist[location], key=lambda x : float(x[0]))[-1*hist_size:]
        return param_hist[location]


    # This is the test method that should accompany batch_tuning
    def batch_test(self, parameters, location):
        returnable = {}
        data_sets = {}
        for k in os.listdir(location):
            
            self.load_data(location + '/' + k, mk_out=False)
            data_sets[k] = self.data
        start = time.perf_counter()
        for k in data_sets:
            self.data = data_sets[k]
            for each in self.data:
                returnable[each] = self.test_fold(each, self.labels, clargs=parameters)

        self.time = time.perf_counter() - start
        self.cl_title = self.fp[-5:-1] +  time.strftime("%d-%m-%Y-%H%M%S", time.localtime())
        self.expdir = self.fp.split('/')[-3] + '-results/' if self.fp.endswith('/') else self.fp.split('/')[-1]
        # self.expdir = self.cl_title + '-results/' 
        try:
            os.mkdir(self.expdir)
        except:
            print('B: Failed to make output directory, assuming it exists already')

        return returnable
                

    # Prints the confusion matrices to self.expdir/self.title-conf-matrices.txt
    # This is a janky method that should be redone in a different way
    # Ideally not in some all in one object either. That seems like an incredibly foolish decision.
    # The thing handling the printing to this should be much more robust, and capable of dealing with a
    # greater possible set of outputs, as it stands right now, it just sorta pukes out a turd.
    # Produces a number 
    #
    # TODO: include the following metrics:
    #       Time
    #       I dont know what fucking else. This is killing me.
    #
    #       
    def print_results(self):
        buff = []
        prebuff = []
        acc_acc = []
        for each in self.matrices:
            buff.append(each + '\n')
            acc = np.sum(np.diag(self.matrices[each]))/float(np.sum(self.matrices[each]))
            buff.append('Fold accuracy : ' + str(acc) + '\n')
            acc_acc.append(acc)
            buff.append(''.join([ '#' for k in range(40)]) + '\n')
            for i in range(len(self.labels)):
                buff.append(','.join([str(k) for k in self.matrices[each][i,] ]) + '\n')
            buff.append(''.join([ '#' for k in range(40)]) + '\n')
        else:
            prebuff.append(self.cl_title + ' Confusion Matrices\n')
            prebuff.append('Generated on :' + time.strftime("%a, %d %b %Y %H:%M:%S \n", time.localtime()))
            prebuff.append('Using the following arguments : ' + ' '.join(sys.argv) + '\n')
            if self.params is not None:
                prebuff.append('Top 5 parameter dictionaries were : ' + ' '.join([str(k) for k in self.params]) + '\n')
                prebuff.append(''.join([ '#' for k in range(40)]) + '\n')
            prebuff.append(self.title + ' Accumulated Confusion Matrix\n')
            summed_matrix = sum([self.matrices[k] for k in self.matrices])
            acc = np.sum(np.diag(summed_matrix))/float(np.sum(summed_matrix))
            prebuff.append('Accuracy across summed folds : ' + str(acc) + '\n')
            av_acc = sum(acc_acc)/len(acc_acc)
            prebuff.append('Avg Cross Fold Accuracy : ' + str(av_acc) + '\n')
            sd_acc = sqrt(sum([(i - av_acc)**2 for i in acc_acc]) / len(acc_acc))
            prebuff.append('Cross Fold Accuracy standard deviation : ' + str(sd_acc) + '\n')
            preds = []
            actual = []
            pred_sum = {}
            for x, i in enumerate(summed_matrix):
                for y, j in enumerate(i):
                    preds = preds + [x for k in range(j)]
                    actual = actual + [y for k in range(j)]
            stats = precision_recall_fscore_support(actual, preds)
            specificity = [] 
            support = [] 
            for x, i in enumerate(summed_matrix):
                negatives = np.sum(summed_matrix) - np.sum(summed_matrix[:,x])
                true_negatives = negatives - np.sum(i) + summed_matrix[x,x]
                specificity.append(true_negatives / negatives)
                support.append(np.sum(summed_matrix[:,x]))
            else: 
                specificity_avg = sum(specificity) / (x + 1)
                specificity_sd = sqrt(sum([(i - specificity_avg)**2 for i in specificity]) / len(specificity))
                support_avg = sum(support) / (x + 1)
                support_sd = sqrt(sum([(i - support_avg)**2 for i in support]) / len(support))
            precision_avg = sum(stats[0]) / len(stats[0])
            precision_sd= sqrt(sum([(i - precision_avg)**2 for i in stats[0]]) / len(stats[0]))
            recall_avg = sum(stats[1]) / len(stats[1])
            recall_sd= sqrt(sum([(i - recall_avg)**2 for i in stats[1]]) / len(stats[1]))
            f1_avg  = sum(stats[2]) / len(stats[2])
            f1_sd= sqrt(sum([(i - f1_avg)**2 for i in stats[2]]) / len(stats[2]))
            
            prebuff.append('Test Time : ' + str(self.time) + '\n')
            prebuff.append('Average AUC : ' + str(self.roc_avg) + '\n')
            prebuff.append('Specificity (avg): ' + str(specificity_avg) + '\n')
            prebuff.append('Specificity (sd): ' + str(specificity_sd) + '\n')
            prebuff.append('Support (avg): ' + str(support_avg) + '\n')
            prebuff.append('Support (sd): ' + str(support_sd) + '\n')
            prebuff.append('Precision (avg): ' + str(precision_avg) + '\n')
            prebuff.append('Precision (sd): ' + str(precision_sd) + '\n')
            prebuff.append('Recall (avg): ' + str(recall_avg) + '\n')
            prebuff.append('Recall (sd): ' + str(recall_sd) + '\n')
            prebuff.append('F1 (avg): ' + str(f1_avg) + '\n')
            prebuff.append('F1 (sd): ' + str(f1_sd) + '\n')
            prebuff.append(''.join([ '#' for k in range(40)]) + '\n')
            if self.act_labels is not None:
                prebuff.append(','.join([str(i) for i in self.act_labels]) + ' labels tested' + '\n')
            for i in range(len(self.labels)):
                prebuff.append(','.join([str(k) for k in summed_matrix[i,] ]) + '\n')
            prebuff.append(''.join([ '#' for k in range(40)]) + '\n')
            buff = prebuff + buff
        with open(self.expdir+ '/' + self.cl_title + '-conf-matrices.txt', 'w') as f:
            for each in buff:
                f.write(each)

    
    # ,Param1,Param2,Accuracy1_total,acc1_std,Accuracy2_percentage,acc2_std,AUC,auc_std,Runtime,rt_std,Precision,prec_std,Recall,rec_std,F1-Score,f1-std,Specificity,spec-std,Support,supp-std

# Standard boiler plate, to run some experiments, based on parameters passed from the command line
if __name__ == '__main__':
    #Parsing arguments
    arg_parser = argparse.ArgumentParser(formatter_class= argparse.RawTextHelpFormatter)
    arg_parser.add_argument('-t', help='Tuning set, used for tuning simple datasets')
    arg_parser.add_argument('-B', help="Whether to use batch tuning or not\n"
                                        "Will later turn into more sensible directory traversal\n"
                                        "this current scheme is bad\n"
                                        "--top_directory\n"
                                        "----child_directory1\n"
                                        "-------testfile1.txt \n"
                                        "-------testfile2.txt \n"
                                        "-------testfile3.txt \n"
                                        "-------testfile4.txt \n"
                                        "-------testfile5.txt \n"
                                        "-------testfile6.txt \n"
                                        "----child_directory2\n"
                                        "-------testfile1.txt \n"
                                        "-------testfile2.txt \n"
                                        "-------testfile3.txt \n"
                                        "-------testfile4.txt \n"
                                        "-------testfile5.txt \n"
                                        "-------testfile6.txt \n"
                                        "...\n", action="store_true")

    arg_parser.add_argument('-c', help='Classifier type', default="GNB")
    arg_parser.add_argument('-p', help='parameter search area key, set in end of file')
    arg_parser.add_argument('-f', help='testing directory')
    a = arg_parser.parse_args()

    # Search area configuration

    args  = { 
            None : None,
            'rf-params' : {'n_estimators': [i for i in range(5, 20, 5)], 'criterion':['gini', 'entropy']},
            'en-params' : {'loss' : ['log'], 'penalty' : ['elasticnet'], 'l1_ratio': [i/100 for i in range(100)]},
            # 'knn-params' : {'n_neighbors' : [x for x in range(5, 1200, 5)] },
            'knn-params' : {'n_neighbors' : [x for x in range(5, 10, 5)] },
            # Cutting down on search space for these svc. By design, the Experiment object will produce
            # every possible combination of the arguments provided on instantiation. Additionally, the
            # SVC classifier will not be affected by changing the certain arguments while other arguments
            # are present. By separating out parameters in this stage, redundancy created by the naive combinations
            # will be eliminated, and the computation time drastically reduced. However, this will create
            # the task of comparing the best of these experiments as a final step.
            #
            # Example:
            #
            #   The following two instantiations will produce the same classifier, 
            #
            #   $cl = SVC(kernel='linear', degree=6, random_state=2)
            #   $cl = SVC(kernel='linear', degree=2, random_state=2)
            #   
            # 
            'svc-params' : {'C' : [.1,.2,.3,.4,.5,.6,.7,.9], 'gamma' : [.001,.01,.1,1,10,100], 'probability' : [True]},
            'svc1-params' : {'C' : [.5], 'gamma' : [10], 'probability' : [True]},
            'svc-blank-params' : { 'C' : [], 
                             'kernel' : [], 
                             'degree' : [], 
                             'gamma' : [], 
                             'coef0' : [], 
                             'probability' : [True], 
                             'decision_function_shape' : ['ovr']}}
    classifiers = { 
            'LDA' : LinearDiscriminantAnalysis,
            'RF' : RandomForestClassifier,
            'SVC' : SVC,
            'EN' : SGDClassifier,
            'MLP' : MLPClassifier,
            'KNN' : KNeighborsClassifier,
            'GNB' : GaussianNB}


    one_gesture = [34]
    four_gesture = [34, 35, 41, 42]
    nine_gesture = [34, 35, 36, 41, 42, 43, 44, 45, 46]
    thirteen_gesture = [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
    twentyfive_gesture = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 25, 32, 33, 34, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46]
    fortyseven_gesture = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]

    # a = Experiment(a.f, classifier=classifiers[a.c], batch=a.B, search_area=args[a.p], tune_loc=a.t, labels=[1,5,46])
    a = Experiment(a.f, classifier=classifiers[a.c], batch=a.B, search_area=args[a.p], tune_loc=a.t, labels=four_gesture, tune=True)

