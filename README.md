# Pytorch implementation of the chamfer distance.

Include a *CUDA version*, and a *python version* with pytorch standard operations.

### CUDA VERSION

`python setup.py install`

Supports multi-gpu
Only support dimension 3. See other branches for dim 2 and 3.

### Python Version

Supports any dimension

### Usage

```python
import dist_chamfer_idx as ext
distChamfer = ext.chamferDist()
p1 = torch.rand(4, 100, 3).cuda()
p2 = torch.rand(4, 200, 3).cuda()
points1 = Variable(p1, requires_grad=True)
points2 = Variable(p2)
dist1, dist2, idx1, idx2= distChamfer(points1, points2)


```

### Benchmark: 1000 * [forward + backward] pass

* Input1 : 32 x 2000 x 3
* Input2 : 32 x 1000 x 3

|  | Timing (sec)    | Memory (GB)     |
| ---------- | -------- | ------- |
| **Cuda**     | **4** | **0.5** |
| **Python**     | 30 | 1  |


### What is the chamfer distance ? 


From [Stanford course](http://graphics.stanford.edu/courses/cs468-17-spring/LectureSlides/L14%20-%203d%20deep%20learning%20on%20point%20cloud%20representation%20(analysis).pdf) on 3D deep Learning
### Aknowledgment 

Original backbone from [Fei Xia](https://github.com/fxia22/pointGAN/blob/master/nndistance/src/nnd_cuda.cu).

#### TODO:

* Discuss behaviour of torch.min() and tensor.min() which causes issues in some pytorch versions

