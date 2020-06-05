import numpy as np
from matplotlib import pyplot as plt
import math
import time
from cmac import CMAC

class DiscreteCMAC(CMAC):

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
    '''
    def process_association_map(self, data, input_min, input_max):
        num_of_assoc_vec = self.num_weights + 1 - self.gen_factor
        for ind, data_val in enumerate(data):
            assoc_vec_ind = self.get_assoc_vec_ind(data_val[0], num_of_assoc_vec, input_min, input_max)
            self.association_map[data_val[0]] = int(math.floor(assoc_vec_ind))

    
    
    def train(self, data, input_min, input_max, total_epoch = 10000, learning_rate = 0.01):
        self.weight_vec = np.ones(self.num_weights) 
        input_X = np.linspace(input_min, input_max, self.num_weights + 1 - self.gen_factor)
        alpha = learning_rate
        epoch = 0

        self.association_map = {}
        self.process_association_map(data, input_min, input_max)

        is_converged = False
        prev_error = 0
        new_error = 0
        start_time = time.clock()
        while epoch <= total_epoch and not is_converged:
            prev_error = new_error
            for ind, data_val in enumerate(data):
                asc_wt_ind = self.association_map[data_val[0]]
                y_output = np.sum(self.weight_vec[asc_wt_ind: asc_wt_ind + self.gen_factor])
                error = data_val[1] - y_output
                correction = alpha*error/self.gen_factor
                self.weight_vec[asc_wt_ind: (asc_wt_ind + self.gen_factor)] = [(self.weight_vec[ind] + correction) for ind in range(asc_wt_ind, (asc_wt_ind + self.gen_factor))]
            
            
            intermediate_output, accuracy = self.predict(data, input_min, input_max, False)
            new_error = 1 - accuracy
            if np.abs(prev_error - new_error) < 0.0000001:
                is_converged = True
            
            epoch = epoch + 1
            print("Discrete CMAC: ", " => g -", self.gen_factor ,"epoch - ", epoch, "  => error - ", new_error, " accuracy - ", accuracy*100)
        
        end_time = time.clock()
        process_time = end_time - start_time
        return process_time, np.copy(self.weight_vec)


    
    def calculate_error(self, y_predict, y_expected):
        error = np.subtract(y_expected, y_predict)
        error_sq = np.power(error, 2)
        sum_error_sq = np.sum(error_sq)
        total_error = np.sqrt(sum_error_sq)/len(y_expected)
        return total_error
    
    def predict(self, data, input_min, input_max, re_process_asso_map = True):
        output = []
        if re_process_asso_map:
            self.process_association_map(data, input_min, input_max)

        for ind, data_val in enumerate(data):
            asc_wt_ind = self.association_map[data_val[0]]
            y_output = np.sum(self.weight_vec[asc_wt_ind: asc_wt_ind + self.gen_factor])
            output.append(y_output)
        
        error = self.calculate_error(output, data[:, 1])
        accuracy = 1 - error
        return output, accuracy


