import numpy as np
from matplotlib import pyplot as plt
import math
from discrete_cmac import DiscreteCMAC

from continuous_cmac import ContinuousCMAC

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

    gen_factor_list = range(1, num_weights)

    convergence_time_discrete = []
    accuracies_discrete = []

    convergence_time_continuous = []
    accuracies_continuous = []

    discrete_cmac = DiscreteCMAC(1, num_weights)
    continuous_cmac = ContinuousCMAC(1, num_weights)

    for gen_factor in gen_factor_list:
        discrete_cmac.set_gen_factor(gen_factor)
        process_time_discrete, weight_vec_discrete = discrete_cmac.train(train_data, 0, TWO_PI, 10000)
        y_output_discrete, accuracy_discrete = discrete_cmac.predict(test_data, 0, TWO_PI)

        convergence_time_discrete.append(process_time_discrete)
        accuracies_discrete.append(accuracy_discrete)

        continuous_cmac.set_gen_factor(gen_factor)
        process_time_continuous, weight_vec_continuous = continuous_cmac.train(train_data, 0, TWO_PI, 10000)
        y_output_continuous, accuracy_continuous = continuous_cmac.predict(test_data, 0, TWO_PI)

        convergence_time_continuous.append(process_time_continuous)
        accuracies_continuous.append(accuracy_continuous)


    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.style.use('seaborn-whitegrid')

    ax1.set_title('Convergence time v/s Generalization Factor')
    ax1.plot(gen_factor_list, convergence_time_discrete, label="Discrete CMAC")
    ax1.plot(gen_factor_list, convergence_time_continuous, color = (1,0,0), label="Continuous CMAC") 
    ax1.set(xlabel='generalization factor', ylabel='Convergence time')
    ax1.legend()
    

    ax2.set_title('Accuracy v/s Generalization Factor')
    ax2.plot(gen_factor_list, accuracies_discrete, label="Discrete CMAC")
    ax2.plot(gen_factor_list, accuracies_continuous, color = (1,0,0), label="Continuous CMAC") 
    ax2.set(xlabel='generalization factor', ylabel='Accuracy')
    ax2.legend()

    plt.show()
 

