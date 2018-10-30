# CNN Fine-tuning

CNN Fine-tuning with Keras.

Usage:
```bash
python fine_tune.py \
    -basenet xception (inception_v3 ...) \
    -dataset /path/to/dataset/ \
    -run_dir /path/to/run/ \
    -name project_name \
    ...
```

TensorBoard:
```bash
tensorboard --logdir /path/to/run/tensorboard/(project_name/)
```
