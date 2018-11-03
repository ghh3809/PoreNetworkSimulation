## PoreNetworkSimulation

:cn:[中文README请点击这里](https://github.com/ghh3809/PoreNetworkSimulation/blob/master/README_CH.md):cn:

### Update Log

V1.0.0

1. Basic pore-network model seepage simulation and dispersion simulation.
2. Change calculation of unit size from multi-attemption method to equation-solving method, and accelerate the construction speed.

### 1. Introduction

The main problem to solve in this project is: to simulate the gas transportation in porous media according to **Effective Pore Network Model**.

```
The effective pore network is a model to describe soil media. It consists of grid-arranged pores to represent pores in real media, and pore throat to connect pores with each other. Differents parameters like pore size, throat size and coordination number can stand for the soil media accordingly.
```

This project is transferred and reconstructed from Matlab to Python.

### 2. Feature

**1. parameterization network**

This project describes the effective pore network according to parameters like pore size, throat size, throat curvature, coordination number etc., and the specific parameterization network is generated.

**2. Randomly construction**

Each parameter will acquire randomness due to the deviation in settings, and various network can be constructed.

**3. Seperating task and data**

Seperate network status and solver from each other, which will bring convinience for recovery of calculation status, and provide basis for multi-thread and distributed calculation.

**4. Improvement of iteration**

Add Jacobi / Gauss-Sedial iteration apart from time step iteration. Without providing time step parameters, the newly added iteration method will converge easier and faster.

**5.Using config file**

Set the parameters in specific config file, rather than change in code.

### 3. Project Structure

```
PoreNetworkSimulation                     ── Project Directory
│
├─ src                                    ── Source Code
│  │
│  ├─ config                              ── Config
│  │  ├─ config.ini.sample                ── Config Sample
│  │  └─ config.ini                       ── Your Config File
│  │
│  ├─ data                                ── Data
│  │  ├─ seepage_<name>_cache.obj         ── Data of Seepage Caching
│  │  ├─ seepage_<name>_status.obj        ── Data of Network Status
│  │  ├─ dispersion_<name>_massflux.obj   ── Data of Mass Flux
│  │  ├─ dispersion_<name>_velocity.obj   ── Data of Velocity
│  │  └─ dispersion_<name>_paths.txt      ── Data of Dispersion Paths
│  │
│  ├─ log                                 ── Log
│  │  ├─ seepage_<name>.log               ── Seepage Simulation Log
│  │  └─ dispersion_<name>_massflux.obj   ── Dispersion Simulation Log
│  │
│  ├─ entity                              ── Entity
│  │  ├─ GasConstant.py                   ── Gas Parameters Class
│  │  ├─ NetworkStatus.py                 ── Network Status Class
│  │  ├─ NetworkStructure.py              ── Network Structure Class
│  │  ├─ Simulator.py                     ── Simulator Class
│  │  └─ StatusCache.py                   ── Status Cache Class
│  │
│  ├─ model                               ── Simulation Model
│  │  ├─ CalculatePermeability.py         ── Seepage Simulation Model
│  │  └─ Dispersion.py                    ── Dispersion Simulation Model
│  │
│  └─ utils                               ── Utilization
│     └─ Tools.py                         ── Tools Class
│
├─ .idea                                  ── Project Config
│
└─ requirements.txt                       ── Project Requirements
```

### 4. Model Introduction

#### 4.1 Permeability Simulation Model

This model is created to simulate stable flow in peameability test (two opposite boundary is fixed pressure).

Input:

1. network parameters (config: `[network]`)
2. gas parameters (config: `[gas]`)
3. boundary condition and initial condition (config: `[status]`)
4. solver type and settings (config: `[solver]`)
5. iteration setting and finish condition (config: `[iteration]`)
6. (optional) intermediate result (if you want to recover)

Output:

1. Peameability result (if you set `iteration.showPermeability` > 0)
2. Network pressure status (save in `src/data/seepage_<name>_status.obj`)

Structure of `src/data/seepage_<name>_status.obj`:

```
seepage_<name>_status.obj         ── (object)          Network Status
├─ sc                             ── (object)          Network Status Config
│  ├─ boundary_type               ── (array[6])        Boundary Type
│  ├─ boundary_value              ── (array[6])        Boundary Value
│  ├─ initial_type                ── (int)             Initial Type
│  └─ initial_value               ── (array[6])        Initial Value
├─ ns                             ── (object)          Network Structure
│  ├─ nc                          ── (object)          Network Structure Config
│  │  ├─ model_size               ── (array[3])        Model Size
│  │  ├─ character_length         ── (float)           Character Length
│  │  ├─ unit_size                ── (float)           Unit Size
│  │  ├─ radius_params            ── (array[2])        Radius Size Ave & Std
│  │  ├─ curvature                ── (float)           Throat Curvature
│  │  ├─ throat_params            ── (array[2])        Throat Size Ave & Std
│  │  ├─ coor_params              ── (array[2])        Coordination Number Size Ave & Std
│  │  ├─ porosity                 ── (float)           Porosity
│  │  └─ anisotropy               ── (array[3])        Anisotropy
│  ├─ model_size                  ── (array[3])        Model Size
│  ├─ character_length            ── (float)           Character Length
│  ├─ radii                       ── (array[x,y,z])    Pore Radii
│  ├─ throatR                     ── (array[x,y,z,26]) Throat Radii
│  ├─ weight                      ── (array[x,y,z,26]) Throat Weight
│  ├─ unit_size                   ── (float)           Unit Size
│  └─ porosity                    ── (float)           True Porosity
├─ gc                             ── (object)          Gas parameters
│  ├─ M                           ── (float)           Molar Mass
│  ├─ R                           ── (float)           Ideal Gas Constant
│  ├─ T                           ── (float)           Temperature
│  └─ u                           ── (float)           Viscosity
├─ model_size                     ── (array[3])        Model Size
└─ pressure                       ── (array[x,y,z])    Pore Pressure
```

#### 4.2 Dispersion Simulation Model

This model is created to calculate dispersion in network according to simulate each particles random movement. It will produce the paths for each particles.

Input: 

1. dispersion parameters (config: `[dispersion]`)
2. network status calculted by 4.1. (`src/data/seepage_<name>_status.obj`)

Output:

1. Particles position in each time step (save in `src/data/dispersion_<name>_paths.txt`).

Structure of `src/data/dispersion_<name>_paths.txt`:

Linear store of paths (array[particles, total-time-steps, 4]). So it is arranged by:

```
particle1_time1
particle1_x1
particle1_y1
particle1_z1
particle1_time2
particle1_x2
particle1_y2
particle1_z2
...
particle2_time1
particle2_x1
particle2_y1
particle2_z1
...
```

Use `reshape(path-data, particles, total-time-steps, 4)` will be useful when analyzing.

### 5. Running

#### 5.1 Environment Requirement

This project runs under Python 2.7.

1. Install [Python 2.7](https://www.python.org/) from official website.
2. Remember to add your python into PATH.
3. Enter the project directory. Use `pip install -r requirements.txt` to install project requirements.

#### 5.2 Run Permeability Simulation

1. Switch to directory `src/config`, set parameters for simulation in file `config.ini`.
2. Switch to directory `src/model`, run file `CalculatePermeability.py` or use command `python -u CalculatePermeability.py`.
3. The running log will show in screen and store in directory `src/log/seepage_<name>.log`. If you choose to save intermediate results in config file, the data will temperately store in `src/data/seepage_<name>_status_<step>.obj`.
4. When the pressure change in one step is less than your setting, the program will stop, and network status data will store in `src/data/seepage_<name>_status.obj`.

#### 5.3 Run Dispersion Simulation

1. Switch to directory `src/config`, set parameters for simulation in file `config.ini`.
2. Switch to directory `src/model`, run file `Dispersion.py` or use command `python -u Dispersion.py`.
3. The running log will show in screen and store in directory `src/log/dispersion_<name>.log`.
4. The final path result will store in `src/data/dispersion_<name>_paths.txt`.

### 6. Special Remind

#### 6.1 Save in Permeability Simulation

When running permeability simulation, you can choose to save the intermediate and final result, and it is mainly for recover from stop or continue the unfinished calculation. However, there will be a situation like this:

When you run permeability simulation, if you set the parameter `iteration.save != 0`, there will be intermediate results and final results saved in directory. And when you run the program at the second time, the program will automatically find the data file like pattern `src/data/seepage_<name>_status.obj`. As the result, your network will rebuild from the existing file (the same as the earlier one), and the simulation will continue. So if you want to build a new network and restart the simulation, please choose the following tips:

1. Delete the file `src/data/seepage_<name>_status.obj`
2. Change your file name `iteration.fileName`
3. Set parameter `iteration.save = 0` when first simulation, and it will not save any result.