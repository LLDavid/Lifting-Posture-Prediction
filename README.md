# Lifting-Posture-Prediction
Baseline NN/cGAN/cVAE
## Prerequisite
* Python 3
* Torch
## Usage
* Download [weights](https://drive.google.com/file/d/12ZqLL51nxMyF4UIqUbr54lgnjqJMHTY-/view?usp=sharing) to folder 'weights'.
* **For ubuntu or wins terminal:**
1. Clone this repo:\
``` git clone git@github.com:LLDavid/Lifting-Posture-Prediction.git```
2. Run:\
```python test.py --type "baseline" --HHW 1.70 0.45 0.45 0.45 0.45```
* **For python IDE on wins:**
1. Down this repo.
2. Open 'test.py'. Edit the input argument 'type' and 'HHW' as you want:\
```model_test(type="baseline", HHW=[1.67, 0.6,0.6,0.45,0.45])```\
Then run the script.
* **Arguments**
1. 'type' has 3 options: "baseline", "cVAE", "cGAN"
2. 'HHW' denote [subject height, left hand height, right hand height, left hand width, right hand width]\
For more details, please refer to our paper (ongoing).
## Example.
<img src="https://github.com/LLDavid/Lifting-Posture-Prediction/blob/master/images/example.PNG" width="500">

## Selected Hypaer-parameters and Candidates.
<img src="https://github.com/LLDavid/Lifting-Posture-Prediction/blob/master/images/hyperp.png" width="800">
