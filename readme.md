# A Speed Accurate Continuous Attractor Neural Network Model of Head Direction Cells


## Setup

This simulation is written in python 3.7 and needs some additional packages installed to run.
The packages are:

	- numpy, for numerical computations
	- csv, for exporting simulation data
	- matplotlib, for plotting data and simulation
	- texttable, for formatted output
	- sci-kit, for analysis of simulation

They can all be installed with the package manager pip. Try

py -m pip install packagename
```python
py -m pip install packagename
```

## Running the Simulation

Before running the simulation, you should set the parameters accordingly. I recommend only changing orders, filename and the verbose flag.
These are all given to the ```simulation()``` function at the bottom of simulation.py.
A sequence of commands can be set through the orders parameter. Here the time interval and the rotation firing rates have to be set as follows:
```[order_1, order_2, ...]``` where ```order_i``` is specified as ```[begin, end, r_0^{ROT}, r_1^{ROT}]```.
Alternatively a predifined sequence of orders can be loaded through either ```experiment()``` or ```experiment_const_speed()```.

Specify a filename through the variable ```exportFileName```, something similar to `simulation_export.csv`.
Additonaly a `verbose` flag can be set to `True` to get a visual representation of the simulation in real time.

Then the simulation can be run with
```python
py simulation.py
```

# Running the Analysis

With the `analysis.py` file the simulation can be anaylsed. The exported `.csv` file can be loaded with `analyse(filename.csv)` at the bottom of the script.
Inside the `analyse()` function further functions can be uncommented to show more plots of the performed simulation.


Then analysis can be run with
```python
py analysis.py
```
