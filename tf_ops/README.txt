### 4. Compile TF Ops
We need to build TF kernels in `tf_ops`. First, activate the virtualenv and make
sure TF can be found with current python. The following line shall run without
error.

```shell
python -c "import tensorflow as tf"
```

Then build TF ops. You'll need CUDA and CMake 3.8+.

```shell
cd tf_ops
mkdir build
cd build
cmake ..
make
```

After compilation the following `.so` files shall be in the `build` directory.

```shell
Open3D-PointNet2-Semantic3D/tf_ops/build
├── libtf_grouping.so
├── libtf_interpolate.so
├── libtf_sampling.so
├── ...
```

Verify that that the TF kernels are working by running

```shell
cd .. # Now we're at Open3D-PointNet2-Semantic3D/tf_ops
python test_tf_ops.py
```

PS: need to install many libs
    need to link tensorflow_framwork
    need to modify code about tf1 to tf2