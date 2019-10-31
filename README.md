# Pytorch Chamfer Distance.

Include a **CUDA** version, and a **PYTHON** version with pytorch standard operations.

### CUDA VERSION

Optionally compile with `python setup.py install` or don't : it will use JIT.

Supports multi-gpu
Build for 3D point clouds. See other repo branches for  2D and 5D point clouds.

### Python Version

Supports any dimension

### Usage

```python
import torch
import dist_chamfer_idx as ext
distChamfer = ext.chamferDist()
p1 = torch.rand(32, 1000, 3).cuda()
p2 = torch.rand(32, 2000, 3).cuda()
points1 = Variable(p1, requires_grad=True)
points2 = Variable(p2)
dist1, dist2, idx1, idx2= distChamfer(points1, points2)
```



### Add it to your project as a submodule

```shell
git submodule add https://github.com/ThibaultGROUEIX/ChamferDistancePytorch
```



### Benchmark: 1000 * [forward + backward] pass

* p1 : 32 x 2000 x 3
* p2 : 32 x 1000 x 3

|  | Timing (sec)    | Memory (GB)     |
| ---------- | -------- | ------- |
| **Cuda**     | **4** | **0.5** |
| **Python**     | 30 | 1  |


### What is the chamfer distance ? 

[Stanford course](http://graphics.stanford.edu/courses/cs468-17-spring/LectureSlides/L14%20-%203d%20deep%20learning%20on%20point%20cloud%20representation%20(analysis).pdf) on 3D deep Learning

### Aknowledgment 

Original backbone from [Fei Xia](https://github.com/fxia22/pointGAN/blob/master/nndistance/src/nnd_cuda.cu).

JIT cool trick from [Christian Miller](https://github.com/chrdiller)

### Troubleshoot:

- `undefined symbol: Zxxxxxxxxxxxxxxxxx `:

--> Fix: Make sure to `import torch` before you `import chamfer`.

#### TODO:

* Discuss behaviour of torch.min() and tensor.min() which causes issues in some pytorch versions
