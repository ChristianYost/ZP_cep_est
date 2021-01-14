
# Differential Cepstrum Zero/Pole Extraction  
This is a python implementation of a new technique for extracting the zeros and poles of a system. We domonstrate its effictiveness for both the analytic expression for the system as well as from the system impulse reponse.  
  
## Setup  
This project was written using python 3.7 and fairly standard packages. To make sure you have the correct packages, check the `requirements.txt` file. If you are creating a new environment, simply create the environment, activate it, and then run   
  
`pip install -r requirements.txt`  
  
## Run  
In order to run the program simply go to the command line and run `python3 main.py`. We have the following command line arguments  
  
`--plot` : flag to plot output  
`--window` : flag to window impulse response. **you don't want to window. To be removed**  
`--N` : discrete spectrum size  
`--noise` : flag to add noise to impulse response  
`--numzeros` : number of zeros in system  
`--numpoles` : number of poles in system  
`--seed` : seed random number processes
    
By specifying the number of zeros and poles, we randomly place them around the top half of the unit circle as well as their corresponding conjugates. `python3 main.py --seed 36 --plot` gives us the following output.  

![output](https://github.com/ChristianYost/ZP_cep_est/blob/master/figs_pics/seed_36_output.png)     
 
