# RNN_predict_crack

[![standard-readme compliant](https://github.com/kentzou/RNN_predict_crack.git)

Gru was used to predict crack openings of concrete DAMS

This repository contains:

1. [data] Original data of cracks in longyangxia dam.
2. [core] data_processer.py defines the DataLoader() class for retrieving training data.
		  model.py defines the model () class for building an RNN model
		  utils.py has a Timer() class for timing, a get_file() function for traversing all the files in the folder, and a custom loss function mas_mse()
3. config.json is a configuration file.
4. RNN_predict_crack.ipynb runs file.


## Table of Contents

- [Install](#install)
- [Usage](#usage)
	- [Generator](#generator)
- [Badge](#badge)
- [Example Readmes](#example-readmes)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)


## Install

download [standard-readme compliant](https://github.com/kentzou/RNN_predict_crack.git)
```
$ pip install keras=2.3.1
$ pip install tensorflow-gpu=1.14.0
$ pip install matplotlib=3.1.2
$ pip install notebook=6.0.2
$ pip install numpy=1.17.3
$ pip install pandas=0.25.3
```

