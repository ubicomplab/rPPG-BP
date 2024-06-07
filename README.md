### Code for "Estimating Blood Pressure with a Camera : A Exploratory Study of Ambulatory Patients with Cardiovascular Disease"

### Enviroment Setup and Raw Data Preparation
We use rPPG-Toolbox setup as our experiment enviroment in the paper.
To setup the enviroment, you can follow the instructions in the '<rPPG-Toolbox Github Repo>' : <https://github.com/ubicomplab/rPPG-Toolbox>
### Reproducing the results
To reproduce our results, 
First, go to the `cd ./ppg_bp` folder.

Set configurations in the `preprocess.sh` script 

run 
```
bash preprocess.sh.
```

Second, select and modify the configuration path in `train.sh`

Third, run 
```
bash train.sh.
```

Finally, after training finished, modify the test configuration with the saved model path in the train configuration.

then run 
```
bash test.sh
```

to get the test results.
