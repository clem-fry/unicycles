![alt text](https://github.com/clem-fry/unicycles/blob/main/media/animation.gif)

## Simulating the RON Unicycle Model for a Reservoir Computer

Random Oscillators Network for Time Series Processing: https://proceedings.mlr.press/v238/ceni24a.html

This is simulations of the Random Oscillators Network (RON) system that acts as a reservoir computer, as created by EIC EMERGE project. Each node is connected to the others using springs and pertubated by the input system. 

The resulting state can be used to predict the next step of a time series, as shown below. 

## PHRESCO Morphological Computation Competition

The goal is to enter the PHRESCO Morphological Computation Competition: https://www.morphologicalcomputation.org/phresco

The goal is to predict the NARAM time series.

![alt text](https://github.com/clem-fry/unicycles/blob/main/media/pipeline_diagram.png)

And the current high-fidelity results are looking promising when simulated in gazebo for the DOTS system.

![alt text](https://github.com/clem-fry/unicycles/blob/main/media/unicycles_graph1.png)

![alt text](https://github.com/clem-fry/unicycles/blob/main/media/gazebo-image.png)


## Implementing on Physical DOTS Robots

The next phase of the project is to run the system on the physical robots. Hopefully, the physical system will be effective in allowing time-series predictions from physical behaviours! 
The bots are only communicating through local interactions, there is no central control coordinating the movements. Stay tuned!

![alt text](https://github.com/clem-fry/unicycles/blob/main/media/dots.jpeg)

## To Run the Low-Fidelity Simulation

Run simulation-swarm.py to create an npz file of the simulation data.
Run data-processing.py to import the npz file, train on the time-series and evaluate the model.
