# TinyVGG
This repository provides Python code to show TinyVGG. You may train it yourself. An inference file is provided to make testing the model after training easier. 

TinyVGG is from https://github.com/poloclub/cnn-explainer/tree/master/tiny-vgg, and used in https://poloclub.github.io/cnn-explainer/.

# Installing
A yaml file with an environment configuration is provided.
```bash
conda env create -f environment.yaml
```

# Training
You must unzip `data.zip`, then you should be able to invoke `python tiny-vgg.py` while your environment is active.