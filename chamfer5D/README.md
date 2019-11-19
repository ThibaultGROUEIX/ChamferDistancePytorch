# Pytorch implementation of the chamfer distance.

Include a *CUDA version*, and a *python version* with pytorch ops.

### CUDA VERSION

`python setup.py install`

Only support dimension 5

### Python Version

Supports any dimension

#### Comparison (1 [forward + backward])

* Input1 : 32 x 244*244 x 3

* Input2 : 32 x 244*244 x 3

* Cuda -> 1 seconds, 700MB

* Python -> memory failure



#### TODO:

* Discuss behaviour of torch.min() and tensor.min() which causes issues in some pytorch versions
* Make it multi-GPU

