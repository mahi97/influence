# Influence

## Requirements

* Python 3 (it might work with Python 2, but I didn't test it)
* [PyTorch](http://pytorch.org/)
* [Stable baselines3](https://github.com/DLR-RM/stable-baselines3)

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Other requirements
pip install -r requirements.txt
```


## Visualization

In order to visualize the results use ```visualize.ipynb```.


## Training

#### PPO

```bash
python main.py --env-name "Reacher-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits
```

## Enjoy

Load a pretrained model from [my Google Drive](https://drive.google.com/open?id=0Bw49qC_cgohKS3k2OWpyMWdzYkk).

Also pretrained models for other games are available on request. Send me an email or create an issue, and I will upload it.

Disclaimer: I might have used different hyper-parameters to train these models.

### MuJoCo

```bash
python enjoy.py --load-dir trained_models/ppo --env-name "Reacher-v2"
```

## Acknowledgements

The PPO implementation is largely based on Ilya Kostrikov's excellent implementation (https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)