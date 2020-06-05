import numpy as np
from matplotlib import pyplot as plt
import math
from discrete_cmac import DiscreteCMAC

from continuous_cmac import ContinuousCMAC

def sort_data(x_data, y_data):
    ind = np.array(x_data).argsort()
    x_sort = x_data[ind]
    y_sort = [y_data[i] for i in ind]
    return x_sort, y_sort

if __name__ == "__main__":

    TWO_PI = 2*np.pi
    sample_input = np.linspace(0, TWO_PI, 100)
    sample_output = np.sin(sample_input)

    total_data = np.column_stack((sample_input, sample_output))
    np.random.shuffle(total_data)

    train_data = total_data[:70]
    test_data = total_data[70:]

    x_train = train_data[:, 0]
    x_test = test_data[:, 0]

    y_train = train_data[:, 1]
    y_test = test_data[:, 1] 

    num_weights = 35
    gen_factor = 10

    ## Discrete CMAC training
    discrete_cmac = DiscreteCMAC(gen_factor, num_weights)
    process_time_discrete, weight_vec_discrete = discrete_cmac.train(train_data, 0, TWO_PI, 10000)
    y_output_discrete, accuracy_discrete = discrete_cmac.predict(test_data, 0, TWO_PI)
    x_discrete_sort, y_discrete_sort = sort_data(x_test, y_output_discrete)

    ## Continuous CMAC training
    continuous_cmac = ContinuousCMAC(gen_factor, num_weights)
    process_time_continuous, weight_vec_continuous = continuous_cmac.train(train_data, 0, TWO_PI, 10000)
    y_output_continuous, accuracy_continuous = continuous_cmac.predict(test_data, 0, TWO_PI)
    x_continuous_sort, y_continuous_sort = sort_data(x_test, y_output_continuous)


    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.style.use('seaborn-whitegrid')

    ax1.set_title('Discrete CMAC Training - gen_factor=10')
    ax1.plot(sample_input, sample_output, label="Original Curve")
    ax1.plot(x_discrete_sort, y_discrete_sort, color = (1,0,0), label="Discrete CMAC curve") 
    ax1.set(xlabel='x-axis', ylabel='y-axis')
    ax1.legend()
    

    ax2.set_title('Continuous CMAC Training - gen_factor=10')
    ax2.plot(sample_input, sample_output, label="Original Curve")
    ax2.plot(x_continuous_sort, y_continuous_sort, color = (1,0,0), label="Continuous CMAC curve")
    ax2.set(xlabel='x-axis', ylabel='y-axis')
    ax2.legend()

    plt.show()





