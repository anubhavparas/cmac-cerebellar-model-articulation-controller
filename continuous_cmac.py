import numpy as np
from matplotlib import pyplot as plt
import math
from cmac import CMAC
import time

class ContinuousCMAC(CMAC):

    def __init__(self, gen_factor, num_weights):
        CMAC.__init__(self, gen_factor, num_weights)

    '''
        >> This will return the index (or the ith sample), in the association_input_vec values, corresponding the actual input value
            |min_value|--|--|--|--|--|--|--|--|max_value| (number of samples from min to max  = number association_input values),
            now we need to check proportionately in which bucket/index the input value will lie
    '''
    def get_assoc_vec_ind(self, input_val, num_of_assoc_vec, input_min, input_max):
        if input_val < input_min:
            return 1
        elif input_val > input_max:
            return num_of_assoc_vec-1
        else:
            proportion_ind = (num_of_assoc_vec-2) * ((input_val - input_min)/(input_max - input_min)) + 1
            return proportion_ind

    
    
    '''
        >> Association map in CMAC is an assiociation unit matrix of size (m x n),
        where m = number of association input vectors (possible values of the sample input)
        n = number of weights
        >> Each row of this matrix has all zeros except range of consecutive cells having a value 1. The number of cells having the value one = gen_factor
            > This represents the weights that will be activated for a specific sample input value.
            > 
                # 1 1 1 0 0 0 0
                # 0 1 1 1 0 0 0
                # 0 0 1 1 1 0 0
                # 0 0 0 1 1 1 0
                # 0 0 0 0 1 1 1  (5 x 7)
            Here gen_factor = 3, num_weights = 7
        
        >> This function will return a map with: 
                key = specific sample input value 
                value = index of the first cell in the range of consecutive ones in that row (corresponding to the input value) 
        >> For continuous CMAC we might need to consider some portion of adjacent windows too for consecutive input values.
        >> To consider the window of adjacent weights, this function return a set of two indices where the sample input value lies
            e.g. if the value = 2.5564 and it lies between the samples 2.4 (at ind=3) and 2.9(at ind=4) (values in the assoc_input_vector), 
                    then this function will return 2.5566 -> (3,4)  
    '''
    def process_association_map(self, data, input_min, input_max):
        num_of_assoc_vec = self.num_weights + 1 - self.gen_factor
        for ind, data_val in enumerate(data):
            assoc_vec_ind = self.get_assoc_vec_ind(data_val[0], num_of_assoc_vec, input_min, input_max)
            assoc_vec_ind_flr = int(math.floor(assoc_vec_ind))
            assoc_vec_ind_ceil = int(math.ceil(assoc_vec_ind))
            
            if (assoc_vec_ind_flr != assoc_vec_ind_ceil):
                self.association_map[data_val[0]] = (assoc_vec_ind_flr, assoc_vec_ind_ceil)
            else:
                self.association_map[data_val[0]] = (assoc_vec_ind_flr, 0)
    
    
    def train(self, train_data, input_min, input_max, total_epoch = 10000, learning_rate = 0.01):
        self.weight_vec = np.ones(self.num_weights) 
        input_X_vec = np.linspace(input_min, input_max, self.num_weights + 1 - self.gen_factor)
        alpha = learning_rate
        epoch = 0

        self.association_map = {}
        self.process_association_map(train_data, input_min, input_max)

        is_converged = False
        prev_error = 0
        new_error = 0
        start_time = time.clock()

        while epoch <= total_epoch and not is_converged:
            prev_error = new_error
            for ind, train_val in enumerate(train_data):
                asc_wt_ind_flr = self.association_map[train_val[0]][0]
                asc_wt_ind_ceil = self.association_map[train_val[0]][1]
                
                left_common = np.abs(input_X_vec[asc_wt_ind_flr] - train_val[0])
                right_common = np.abs(input_X_vec[asc_wt_ind_ceil] - train_val[0])
                
                left_contri_ratio = right_common/(left_common + right_common)
                right_contri_ratio = left_common/(left_common + right_common)
                
                y_output = (left_contri_ratio * np.sum(self.weight_vec[asc_wt_ind_flr: asc_wt_ind_flr + self.gen_factor])) + (right_contri_ratio * np.sum(self.weight_vec[asc_wt_ind_ceil: asc_wt_ind_ceil + self.gen_factor]))
                
                error = train_val[1] - y_output
                correction = alpha*error/self.gen_factor
                
                self.weight_vec[asc_wt_ind_flr: (asc_wt_ind_flr + self.gen_factor)] = \
                                            [(self.weight_vec[ind] + correction) for ind in range(asc_wt_ind_flr, (asc_wt_ind_flr + self.gen_factor))]
                
                self.weight_vec[asc_wt_ind_ceil: (asc_wt_ind_ceil + self.gen_factor)] = \
                                            [(self.weight_vec[ind] + correction) for ind in range(asc_wt_ind_ceil, (asc_wt_ind_ceil + self.gen_factor))]
                
            
            intermediate_output, accuracy = self.predict(train_data, input_min, input_max, False)
            new_error = 1 - accuracy
            if np.abs(prev_error - new_error) < 0.0000001:
                is_converged = True
                
                
            epoch = epoch + 1
            print("Continuous CMAC: ", " => g -", self.gen_factor, "epoch - ", epoch, "  => error - ", new_error, " accuracy - ", accuracy*100)
        
        end_time = time.clock()
        process_time = end_time - start_time
        return process_time, np.copy(self.weight_vec)

    

    def calculate_error(self, y_predict, y_expected):
        error = np.subtract(y_expected, y_predict)
        error_sq = np.power(error, 2)
        sum_error_sq = np.sum(error_sq)
        total_error = np.sqrt(sum_error_sq)/len(y_expected)
        return total_error
    
    
    def predict(self, test_data, input_min, input_max, re_process_asso_map = True):
        output = []

        input_X_vec = np.linspace(input_min, input_max, self.num_weights + 1 - self.gen_factor)

        if re_process_asso_map:
            self.process_association_map(test_data, input_min, input_max)
        
        for ind, test_val in enumerate(test_data):
            asc_wt_ind_flr = self.association_map[test_val[0]][0]
            asc_wt_ind_ceil = self.association_map[test_val[0]][1]

            left_common = np.abs(input_X_vec[asc_wt_ind_flr] - test_val[0])
            right_common = np.abs(input_X_vec[asc_wt_ind_ceil] - test_val[0])

            left_contri_ratio = right_common/(left_common + right_common)
            right_contri_ratio = left_common/(left_common + right_common)

            y_output = (left_contri_ratio * np.sum(self.weight_vec[asc_wt_ind_flr: asc_wt_ind_flr + self.gen_factor])) + (right_contri_ratio * np.sum(self.weight_vec[asc_wt_ind_ceil: asc_wt_ind_ceil + self.gen_factor]))

            output.append(y_output)
        
        error = self.calculate_error(output, test_data[:, 1])
        accuracy = 1 - error
        return output, accuracy


