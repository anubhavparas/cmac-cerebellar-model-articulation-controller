# CMAC - Cerebellar Model Articulation Controller

## This is to program and train a 1-D discrete and continuous CMAC network.
- Effect of overlap area on generalization and time to convergence was explored.
- 35 weights were used for the CMAC network and the function was samples at 100 evenly spaced points.

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






