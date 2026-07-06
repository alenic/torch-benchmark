# torch-benchmark

Simple image pipeline and model throughput benchmarking for PyTorch vision workloads.

## What `benchmark_cv.py` does

`benchmark_cv.py` measures:

- image loading + transform time
- end-to-end training throughput
- end-to-end evaluation throughput

It walks a folder recursively and benchmarks all supported images it finds:
`.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`

The dataset uses a dummy label, so this is meant for performance benchmarking, not accuracy training.

## Quick start

Install the main dependencies in your environment:

```bash
pip install torch torchvision timm pillow opencv-python albumentations
```

Run a training benchmark:

```bash
python3 benchmark_cv.py \
  --root /path/to/images \
  --model resnet50 \
  --batch_size 32 \
  --num_workers 4 \
  --n_images 1000 \
  --device cuda:0
```

Run an evaluation benchmark:

```bash
python3 benchmark_cv.py \
  --root /path/to/images \
  --model resnet50 \
  --batch_size 32 \
  --num_workers 4 \
  --n_images 1000 \
  --device cuda:0 \
  --eval
```

## Image folder layout

Any recursive image folder works. For example:

```text
dataset/
  class_a/
    img1.jpg
    img2.jpg
  class_b/
    img3.png
```

Class names are not used by the benchmark. The script only needs image files under `--root`.

## Useful options

- `--root`: root folder containing images
- `--model`: timm model name, default `resnet50`
- `--batch_size`: batch size, default `8`
- `--num_workers`: dataloader workers, default `0`
- `--loader`: `pil` or `cv2`, default `pil`
- `--img_size`: resize/crop size, default `224`
- `--n_images`: number of images to process before stopping
- `--device`: device string such as `cuda:0` or `cpu`
- `--pin_memory`: enable dataloader pinned memory
- `--eval`: run inference benchmark instead of training benchmark

## Example with OpenCV loader

```bash
python3 benchmark_cv.py \
  --root /path/to/images \
  --loader cv2 \
  --batch_size 32 \
  --num_workers 4 \
  --n_images 1000 \
  --device cuda:0
```

## Output

The script prints a short summary like:

```text
============= Results ===============
phase: Train
model: resnet50
classes: 10
image size: 224
batch size: 32
workers: 4
image loader: pil
device: cuda:0
pin memory: False

Data Time percentage: 12.34 %
Images/sec: 987.65
```

## Notes

- If the dataset is too small for the selected `--batch_size`, the dataloader can be empty because `drop_last=True`.
- `--num_classes` only changes the model head size for the benchmark.
- For GPU benchmarking, use a `--batch_size` that fits in memory.
