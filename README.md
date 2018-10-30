# object_detection_inference
Example of TensorFlow object detection inference.

## Steps

* Install dependencies
```python
sudo pip install numpy tensorflow-gpu pillow
```

* Install Python version OpenCV
```bash
sudo apt-get install python-opencv
```

* Clone the project
```bash
git clone https://github.com/SuJiaKuan/object_detection_inference
```

* Go to the project directory
```bash
cd object_detection_inference
```

* Get a pre-trained model from [TensorFlow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
(In this case, we use [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz))
```bash
# Downalod the model
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
# Untar the tarball
tar zxvf ssd_mobilenet_v1_coco_2017_11_17.tar.gz
```

* Now, we can start running!
```bash
# Usage: python main.py path_to_model src
python main.py ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb demo.mp4
```
