# CMAC - Cerebellar Model Articulation Controller

## This is to program and train a 1-D discrete and continuous CMAC network. 

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






