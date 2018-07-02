import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('inf')
        best_model = None
        
        for n in range (self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(n)    
                logL = hmm_model.score(self.X, self.lengths)
                logN = np.log(len(self.X))
                
                p = n ** 2 + 2 * n * hmm_model.n_features - 1
                BIC = -2 * logL + p * logN

                if BIC < best_score:
                    best_model = hmm_model
                    best_score = BIC
            except:
                pass
            
        if best_model == None:
            return self.base_model(self.n_constant)

        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        
        best_score = float('-inf')
        best_model = None
        
        for n in range (self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(n) 
                logL = hmm_model.score(self.X, self.lengths)
                sum_logL_ni = 0.0
                for word in self.words:
                    if word == self.this_word:    
                        continue
                    
                    x_ni, lengths_ni = self.hwords[word]
                    
                    try:
                        logL_ni = hmm_model.score(x_ni, lengths_ni)
                        sum_logL_ni += logL_ni
                    except:
                        pass
                
                DIC = logL - sum_logL_ni / (len(self.words)-1)
                
                if DIC > best_score:
                    best_model = hmm_model
                    best_score = DIC                    
                
            except:
                pass
            
        if best_model == None:
            return self.base_model(self.n_constant)

        return best_model     
        


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''    
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        score = float('-inf')
        best_n = self.n_constant
        
        if(len(self.sequences) > 1):
            try:
                #print('seq', self.sequences)
                fold = KFold(min(3, len(self.sequences)))
                
                for n in range(self.min_n_components, self.max_n_components + 1):
                    try:
                        log_total = []
                        model_n_score = float('-inf')
                        
                        for cv_train_idx, cv_test_idx in fold.split(self.sequences):
                            train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                            test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
        
                            try:
                                hmm_model = self.base_model(n).fit(train_X, train_lengths)
                                logL = hmm_model.score(test_X, test_lengths)
                                log_total.append(logL)
                                
                                # calcs more often but i don't need to wory about empty scores
                                model_n_score = np.mean(log_total)       
                                
                            except:
                                pass
                            
                        if(model_n_score > score):
                            score = model_n_score
                            best_n = n
                    except:
                        pass
                    
            except:
                pass

                        
        if(len(self.sequences) == 1):
            print ('!!!! ireached this code') # code is not reached in first unit test?
            for n in range(self.min_n_components, self.max_n_components + 1):
                log_total = []
                model_n_score = float('-inf')
                
                try:
                    hmm_model = self.base_model(n).fit(self.X, self.lengths)
                    logL = hmm_model.score(self.X, self.lengths) 
                    
                    if(logL > score):
                        best_n = n
                        score = logL
                    
                except:
                    pass       
        
        return self.base_model(best_n)