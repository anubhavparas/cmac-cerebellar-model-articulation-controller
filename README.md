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

#### Approach:
 1D function: sin(x)
 ##### For Discrete CMAC:
  - 

#### Information to run the code 
- In the terminal where you can run python scripts go to the directory where these files are,
- Make sure that numpy is installed.

- To run the code for training the models (discrete and continuous cmac):

  **$ python cmac_nn.py**

- To run the code for checking the effect of generalization factor or overlapping on the convergence and the accuracy of the models:

  **$ python cmac_convergence.py**


Results:

Discrete CMAC - generalization factor = 10
![alt text](./images/d_cmac_10.PNG?raw=true "Discrete CMAC")

Continuous CMAC - generalization factor = 10
![alt text](./images/c_cmac_10.PNG?raw=true "Continuous CMAC")






