Simulation codes and data for ``Bayesian estimation of coupling strength and heterogeneity in a coupled oscillator model from macroscopic observations'' by Y Kato, S Kashiwamura, E Watanabe, M Okada, and H Kori.

When writing this README.md file, the author (Yusuke Kato) consulted the README.md file in the following GitHub repository: https://github.com/ishiihidemasa/24-coupling-facilitate-impede-escape. 

- The simulation environment can be reproduced by Docker or Apptainer. 
- For the brief tutorial of these softwares, see the section "How to use Docker" and "How to use Apptainer". 

# Data generation and EMC simulation
The simulation codes are stored in the `estimate` directory. The C++ environment with necessary packages can be reproduced by Docker (if you use your local environment) or Apptainer (if you use HPCs).

- If you use your local environment
  - After setting the environment with Docker, you can run the code with the following commands:
  ```
  g++ -fopenmp -O3 filename.cpp
  OMP_NUM_THREADS=20 > log.txt ./a.out
  ```

- If you use HPCs
  - After making `filename.sif` with Apptainer,  you can run the code with the following commands: 
  ```
  apptainer exec filename.sif g++ -fopenmp -O3 filename.cpp
  apptainer exec --env OMP_NUM_THREADS=20 filename.sif > log.txt ./a.out
  ```
In either case, you can change the number of threads ("20" in the above commands) depending on your CPU. 

The simulation results in `plot/results` directory were calculated using the scripts named `EMC_something.cpp` on HPC with Apptainer. 
You can use the C++ codes to reproduce the stored data or to perform additional simulations. 

Several notes about the contents in `plot/results` directory: 
- The directory `output_files_something` coppresponds to the simulation results of `EMC_something.cpp`. 
- Note that the two `.txt` files (`OA_rep_seed.txt` and `kuramoto_rep_seed.txt`) were created by `.py` files stored in the `plot` directory. These data are used to determine which random number seed to use in the single EMC simulations (e.g., `EMC_OA_sigma01.cpp`). 

# Plotting the figures
The python environment with necessary packages can be reproduced by Docker.
- The scripts for figure creation are stored in the `plot` directory. Run the scripts named `figX_somthing.py` to reproduce the figures in our article.  
- The necessary data, which are the results of Cpp codes in the `estimate` directory, are stored in `plot/results` directory. These data are loaded automatically by the Python codes in the `plot` directory.  


# How to use Docker
- If you use VS Code ``Dev Containers'' extension and Docker Desktop
  - Download our GitHub repository and open either the `estimate` or `plot` directory as container, using `Dev Containers: Open Folder in Container...` command.
    - You may need to reload the window after container is built.
  - You can execute the Cpp or Python codes in the container.
- If you use Docker Desktop
  - Use `Dockerfile` in either `estimate/.devcontainer` or `plot/.devcontainer` directory to reproduce the environment.

# How to use Apptainer
- Make sure that Apptainer is installed in the HPC. 
- Execute `apptainer build filename.sif filename.def` to make `.sif` file. 

If you have any questions when running the codes, please contact Yusuke Kato.
