# CMAC - Cerebellar Model Articulation Controller

The CMAC was first proposed as a function modeler for robotic controllers by James Albus in
1975 (hence the name), but has been extensively used in reinforcement learning and also as for
automated classification in the machine learning community.

- CMAC computes a function f(x1, x2,.....xn), where n is the number of input dimensions. The input
space is divided up into hyper-rectangles, each of which is associated with a memory cell. The
contents of the memory cells are the weights, which are adjusted during training.
- A change of value of the input point results in a change in the set of activated hyper-rectangles,
and therefore a change in the set of memory cells participating in the CMAC output. The CMAC
output is therefore stored in a distributed fashion, such that the output corresponding to any
point in input space is derived from the value stored in a number of memory cells (hence the
name associative memory). This provides **generalization**.



## This is to program and train a 1-D discrete and continuous CMAC network.

- Effect of overlap area on generalization and time to convergence was explored.
- 35 weights were used for the CMAC network and the function was sampled at 100 evenly spaced points.

### Approach:
 1D function: sin(x)
 ##### For Discrete CMAC:
  - For the total_input values were taken from 0 to 2pi and were sampled at 100 points.
  For the programming the initial 1D CMAC, generalization factor was taken as 10, **gen_factor = 10**.
  - As the **num_weights = 35** and **gen_factor = 10**, we need to find the number of **associative vectors/input values** that will be mapped to the rows of the association unit/matrix.
  - So, **num_assoc_vec = num_weights + 1 – gen_factor**.
  - This is pretty intuitive as each row of the **association matrix** will have specific number of consecutive ones (here, gen_factor) for the overlapping scenario so the association matrix will be of the dimension : **num_assoc_vec x num_weights**.
  - The sample input space was redefined by sampling **‘num_assoc_vec’** values from 0 to 2.pi, where each sample value corresponds to a specific row of association matrix/unit.
  - Refer to this comment snippet from the code:
  ![alt text](./images/comment_snippet_assoc_matrix.jpg?raw=true "Association matrix example")
  - The cells values that are one in each row represent the weights that getting activated for that specific input value:
   ![alt text](./images/assoc_vec_mapping.jpg?raw=true "Association matrix example")
   - From this mapping we can find the weights that will be active for each input in the input space.
   - To train the discrete CMAC, for each input –> get the associated weightd -> sum the weights to get the output value -> compare the output to the expected value and calculate the error -> updated the weights using the formula: 
      **weight<sub>n+1</sub> = weight<sub>n</sub> + error<sub>n</sub> * learning_rate/gen_factor**
   - And the process is repeated until the **difference between the total new_error and the total previous error becomes less than a pre-decided threshold**.
  
  ##### For Continuous CMAC:
  - The process to the train the continuous CMAC was similar to that of the discrete CMAC.
  - The only difference here was to consider the proportion of weights (overlapping window size) to find out which weights are contributing in which proportion.
  - For continuous CMAC we might need to consider some portion of adjacent windows too for consecutive input values.
  - To consider the window of adjacent weights, this function return a set of two indices where the sample input value lies: e.g. if the value = 2.5564 and it lies between the samples 2.4 (at ind=3) and 2.9(at ind=4) (values in the assoc_input_vector), then this function will return 2.5566 -> (3,4)
  - The training process and convergence criteria were same a that for the discrete CMAC.


### Results:

Discrete CMAC - generalization factor = 10
![alt text](./images/d_cmac_10.PNG?raw=true "Discrete CMAC")

Continuous CMAC - generalization factor = 10
![alt text](./images/c_cmac_10.PNG?raw=true "Continuous CMAC")

Comparison of Convergence time with change in generalization factor:
![alt text](./images/conv_time_vs_gen_factor.jpg?raw=true "Convergence time v/s Generalization Factor")

Comparison of Accuracy with change in generalization factor:
![alt text](./images/accuracy_vs_gen_factor.jpg?raw=true "Accuracy v/s Generalization Factor")



### Information to run the code 
- In the terminal where you can run python scripts go to the directory where these files are,
- Make sure that numpy is installed.

- To run the code for training the models (discrete and continuous cmac):

  **$ python cmac_nn.py**

- To run the code for checking the effect of generalization factor or overlapping on the convergence and the accuracy of the models:

  **$ python cmac_convergence.py**





