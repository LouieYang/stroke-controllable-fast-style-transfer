# Stroke Controllable Fast Style Transfer

This repository contains the public release of the Python implementation of stroke controllable fast style transfer.

[**Stroke Controllable Fast Style Transfer with Adaptive Receptive Fields**](https://arxiv.org/abs/1802.07101)

If you use this code or find this work useful for your research, please cite:
```
@article{jing2018stroke,
  title={Stroke Controllable Fast Style Transfer with Adaptive Receptive Fields},
  author={Jing, Yongcheng and Liu, Yang and Yang, Yezhou and Feng, Zunlei and Yu, Yizhou and Tao, Dacheng and Song, Mingli},
  journal={arXiv preprint arXiv:1802.07101},
  year={2018}
}
```

## Getting Started

Implemented and tested on Ubuntu 14.04 with Python 2.7 and Tensorflow 1.4.1.

### Dependencies
* [Tensorflow](https://www.tensorflow.org/)
* [Numpy](www.numpy.org/)
* [Pillow](https://pypi.python.org/pypi/Pillow/)
* [Scipy](https://www.scipy.org/)

### Download pre-trained VGG-19 model
The VGG-19 model of tensorflow is adopted from [VGG Tensorflow](https://github.com/machrisaa/tensorflow-vgg) with few modifications on the class interface. The VGG-19 model weights is stored as .npy file and could be download from [Google Drive](https://drive.google.com/file/d/0BxvKyd83BJjYY01PYi1XQjB5R0E/view?usp=sharing) or [BaiduYun Pan](https://pan.baidu.com/s/1o9weflK). After downloading, copy the weight file to the **/vgg19** directory

## Basic Usage
### Train the network
Use train.py to train a new stroke controllable style transfer network. Run `python train.py -h` to view all the possible parameters. The dataset used for training is MSCOCO train 2014 and could be download from [here](http://cocodataset.org/#download), or we provide a random selected 2k images from MSCOCO. Example usage:

```
$ python train.py \
    --style /path/to/style_image.jpg \
    --train_path /path/to/MSCOCO_dataset \
    --sample_path /path/to/content_image.jpg
```

### Freeze model
Use pack_model.py to freeze the saved checkpoint. Run `python pack_model.py -h` to view all parameter. Example usage:

```
$ python pack_model.py \
    --checkpoint_dir ./examples/checkpoint/some_style \
    --output ./examples/model/some_style.pb
```

### Inference
Use inference_style_transfer.py to inference the content image based on the freezed style model. Set `--interp N` to enable interpolation inference where `N` is the number of the continuous stroke results.

```
$ python inference_style_transfer.py \
    --model ./examples/model/some_style.pb \
    --serial ./examples/serial/default/ \
    --content ./examples/content/some_content.jpg
```

## Examples
Here are some results from our algorithm

### Spatial Stroke Size Control
From left to right are content&style, mask, same stroke size across image result and spatial stroke size control result.
<p align='center'>
    </br>
    <img src='examples/readme_examples/spatial_control/1_c.jpg' width='210'>
    <img src='examples/readme_examples/spatial_control/1_m.png' width='210'>
    <img src='examples/readme_examples/spatial_control/1_256.jpg' width='210'>
    <img src='examples/readme_examples/spatial_control/1_result.jpg' width='210'>
    </br>
    <img src='examples/readme_examples/spatial_control/2_c.jpg' width='210'>
    <img src='examples/readme_examples/spatial_control/2_m.png' width='210'>
    <img src='examples/readme_examples/spatial_control/2_256.jpg' width='210'>
    <img src='examples/readme_examples/spatial_control/2_result.jpg' width='210'>
    </br>
    <img src='examples/readme_examples/spatial_control/3_c.jpg' width='210'>
    <img src='examples/readme_examples/spatial_control/3_m.png' width='210'>
    <img src='examples/readme_examples/spatial_control/3_256.jpg' width='210'>
    <img src='examples/readme_examples/spatial_control/3_result.jpg' width='210'>
    </br>
    <img src='examples/readme_examples/spatial_control/4_c.jpg' width='210'>
    <img src='examples/readme_examples/spatial_control/4_m.png' width='210'>
    <img src='examples/readme_examples/spatial_control/4_256.jpg' width='210'>
    <img src='examples/readme_examples/spatial_control/4_result.jpg' width='210'>
    </br>
    <img src='examples/readme_examples/spatial_control/5_c.jpg' width='210'>
    <img src='examples/readme_examples/spatial_control/5_m.jpg' width='210'>
    <img src='examples/readme_examples/spatial_control/5_256.jpg' width='210'>
    <img src='examples/readme_examples/spatial_control/5_result.jpg' width='210'>
    </br>
</p>


### Continuous Stroke Size Control
Stroke grows from left to right. We zoom in on the same region (red frame) to observe the variations of stroke sizes
<p align='center'>
    </br>
    <img src='examples/readme_examples/continuous_control/1/1_rectangle.jpg' width='140'>
    <img src='examples/readme_examples/continuous_control/1/2_rectangle.jpg' width='140'>
    <img src='examples/readme_examples/continuous_control/1/3_rectangle.jpg' width='140'>
    <img src='examples/readme_examples/continuous_control/1/4_rectangle.jpg' width='140'>
    <img src='examples/readme_examples/continuous_control/1/5_rectangle.jpg' width='140'>
    <img src='examples/readme_examples/continuous_control/1/6_rectangle.jpg' width='140'>
    </br>
    <img src='examples/readme_examples/continuous_control/1/1_detail.png' width='140'>
    <img src='examples/readme_examples/continuous_control/1/2_detail.png' width='140'>
    <img src='examples/readme_examples/continuous_control/1/3_detail.png' width='140'>
    <img src='examples/readme_examples/continuous_control/1/4_detail.png' width='140'>
    <img src='examples/readme_examples/continuous_control/1/5_detail.png' width='140'>
    <img src='examples/readme_examples/continuous_control/1/6_detail.png' width='140'>
    </br>
    <img src='examples/readme_examples/continuous_control/2/1_rectangle.jpg' width='140'>
    <img src='examples/readme_examples/continuous_control/2/2_rectangle.jpg' width='140'>
    <img src='examples/readme_examples/continuous_control/2/3_rectangle.jpg' width='140'>
    <img src='examples/readme_examples/continuous_control/2/4_rectangle.jpg' width='140'>
    <img src='examples/readme_examples/continuous_control/2/5_rectangle.jpg' width='140'>
    <img src='examples/readme_examples/continuous_control/2/6_rectangle.jpg' width='140'>
    </br>
    <img src='examples/readme_examples/continuous_control/2/1_detail.png' width='140'>
    <img src='examples/readme_examples/continuous_control/2/2_detail.png' width='140'>
    <img src='examples/readme_examples/continuous_control/2/3_detail.png' width='140'>
    <img src='examples/readme_examples/continuous_control/2/4_detail.png' width='140'>
    <img src='examples/readme_examples/continuous_control/2/5_detail.png' width='140'>
    <img src='examples/readme_examples/continuous_control/2/6_detail.png' width='140'>
    </br>
    <img src='examples/readme_examples/continuous_control/3/1_rectangle.jpg' width='140'>
    <img src='examples/readme_examples/continuous_control/3/2_rectangle.jpg' width='140'>
    <img src='examples/readme_examples/continuous_control/3/3_rectangle.jpg' width='140'>
    <img src='examples/readme_examples/continuous_control/3/4_rectangle.jpg' width='140'>
    <img src='examples/readme_examples/continuous_control/3/5_rectangle.jpg' width='140'>
    <img src='examples/readme_examples/continuous_control/3/6_rectangle.jpg' width='140'>
    </br>
    <img src='examples/readme_examples/continuous_control/3/1_detail.png' width='140'>
    <img src='examples/readme_examples/continuous_control/3/2_detail.png' width='140'>
    <img src='examples/readme_examples/continuous_control/3/3_detail.png' width='140'>
    <img src='examples/readme_examples/continuous_control/3/4_detail.png' width='140'>
    <img src='examples/readme_examples/continuous_control/3/5_detail.png' width='140'>
    <img src='examples/readme_examples/continuous_control/3/6_detail.png' width='140'>
    </br>
</p>

## Contact

Feel free to contact us if there is any question (Yang Liu lyng_95@zju.edu.cn)
