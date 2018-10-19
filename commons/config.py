#!/usr/bin/python
import sys,os
base_path = os.path.dirname(__file__);


MODEL_NAME = os.path.join(base_path,'../model');
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = MODEL_NAME + '/qipan_label_map.pbtxt'
NUM_CLASSES = 51
