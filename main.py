import os
import sys

import cv2
import numpy as np
import tensorflow as tf

from utils import label_map_util
from utils import visualization_utils as vis_util


# To detect.
MIN_SCORE_THRESH = 0.25

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# Number of label classes
NUM_CLASSES = 90


def norm_color(color):
  norms = []
  for i in range(3):
    norms.append(color[i] / 255.0)
  return (norms[0], norms[1], norms[2])


def main(path_to_model, src):
  # Load a model into memory
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_model, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  # Load label map
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

  category_index = label_map_util.create_category_index(categories)

  cap = cv2.VideoCapture(src)

  if not cap.isOpened():
      print('Failed to open {}'.format(src))
      return

  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      # Definite input and output Tensors for detection_graph
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
      detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      while True:
        # Read image frame by frame
        success, image = cap.read()
        if not success:
          break

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_expanded = np.expand_dims(image, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_expanded})

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
          image,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

        cv2.imshow('Detection', image)
        cv2.waitKey(10)

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  if len(sys.argv) < 3:
    print('Usage: python {} path_to_model src'.format(sys.argv[0]))
    sys.exit(-1)

  path_to_model = sys.argv[1]
  src = sys.argv[2]
  print('Path to model file: {}'.format(path_to_model))
  print('Input video: {}'.format(src))

  main(path_to_model, src)
