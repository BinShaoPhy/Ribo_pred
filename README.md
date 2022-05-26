### riboPred

This program is part of a project that predicts ribosome footprints in bacteria cells using amino acid sequences. It uses a transformer-based machine learning model to predict ribosome density at a particular AA position based on the sequence around it.

Contact: Bin Shao, bshao@fas.harvard.edu

### Requirement
`tensorflow` (tested on version `2.80`)

### Training
```
usage: train.py [-h] [-e EPOCH] [-l LEARNING] [-c NCONV] [-n NHEADS] -o OUT [-verbose]

optional arguments:
  -h, --help            show this help message and exit
  -o OUT, --out OUT     model output directory

Test parameters:
  Parameters for building model

  -e EPOCH, --epoch EPOCH
                        epoch number for model training
  -l LEARNING, --learning LEARNING
                        learning rate for model training
  -c NCONV, --nconv NCONV
                        number of convolution layers
  -n NHEADS, --nheads NHEADS
                        number of attention heads
  -verbose, --verbose   Print training progress
  ```

### Argument

| Argument | Description | Example |
| :---- | :---- | :---- |
| -h --help | show this help message and exit | NA |
| -o --out| output folder name | modeloutput |
| -e --epoch| epoch number for the model training | 100 |
| -l --learning| learning rate for the model training | 0.0001 |
| -c --nconv| number of convolution layers | 5 |
| -n --nheads| number of heads in the multi-head attention layer | 5 |
| -v --verbose| options to display training process | 1 |

### Performance
<sub>Prediction in E coli (DH10b), M9 media/exponential growth phase.</sub>

<img width="260" alt="image" src="https://user-images.githubusercontent.com/98933203/170510551-d194abe3-bcd1-4902-bd58-08c3d0fe77d1.png">


### Reference
Espah Borujeni, A., Zhang, J., Doosthosseini, H. et al. Genetic circuit characterization by inferring RNA polymerase movement and ribosome usage. [Nat Commun 11, 5001 (2020).](https://www.nature.com/articles/s41467-020-18630-2)

