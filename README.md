# Expanding memory in recurrent spiking networks

This repository contains all the neccesary code to test or retrain the delay-based neural network presented in 

[Balafrej, I., Bahadi, S., Rouat, J. and Alibart, F. (2025). Enhancing temporal learning in recurrent spiking networks for neuromorphic applications.](https://arxiv.org/abs/2310.19067).

## Executing the code
For each task, start by going in the corresponding directory. Each directory contains all the neccessary code to test or retrain the model on the specific task. 

All the code was tested with python version 3.8.18 and the dependencies listed in `requirements.txt`.

To test, run:
```bash
python main.py test
```

To train, run:
```bash
python main.py
```

To run hyperparameter optimization (required for most tasks), run this command multiple times, and pick the best resulting network with validation metrics:
```bash
python main.py rnd_main
```

Full CLI documentation is provided with the "--help" command, e.g.:
```bash
python main.py test --help
```

### Cue accumulation benchmark (cue_accumulation_experiment)
Pretrained weights are available in the "weights" subdirectory and will be loaded automatically during testing. 

### Permuted Sequential MNIST (psmnist_experiment)
Pretrained weights are available in the "weights" subdirectory and will be loaded automatically during testing. 

### Delayed Neuromorphic MNIST (dnmnist_experiment)
Pretrained weights are available in the "weights" subdirectory. The weights must be selected during testing:
```bash
python main.py test <experiment_id>
```

### Loihi metrics (loihi_experiment)
This directory presents the cost in latency and energy of having synaptic delays on Loihi. To reproduce results, run:

```bash
python delay.py
```
