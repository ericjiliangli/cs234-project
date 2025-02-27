# cs234-project

Please use the policy network in `policy_network.py` as your base policy network.

To make the BipedalWalker-v3 environment partially observable, as suggested by the TA, please follow the implementation in `partially_observable.ipynb`. That is, right now we choose to mask out all 10 LIDAR features entirely. This is an arbitrary choice, feel free to let everyone know if you find a better masking option that makes the environment harder, but also learnable.