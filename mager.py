#!/usr/bin/python
#-*- coding:utf-8 -*-
import os,sys
#==============================================================
''' import tagpy wordsegs '''
base_path = os.path.dirname(__file__);
sys.path.append(os.path.join(base_path,'./commons'));

#==============================================================
import config,io,re,base64,collections
import numpy as np
import tensorflow as tf
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util


class Mager:
	def __init__(self):
		pass;
	def init(self):
		self.status = None;
		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(config.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
				print("init load model success")
		label_map = label_map_util.load_labelmap(config.PATH_TO_LABELS)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=config.NUM_CLASSES, use_display_name=True)
		self.category_index = label_map_util.create_category_index(categories)


	def check_image(self,base64_str):
		try:
			base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
			byte_data = base64.b64decode(base64_data)
			image = io.BytesIO(byte_data)
			img = Image.open(image)
			return img
		except Exception as e:
			print(e)
			return None

	def image_to_base64(self,image_data):
		output_buffer = io.BytesIO()
		image_data.save(output_buffer, format='JPEG')
		byte_data = output_buffer.getvalue()
		base64_str = base64.b64encode(byte_data)
		return base64_str

	# the array based representation of the image will be used later in order to prepare the
	# result image with boxes and labels on it.
	def load_image_into_numpy_array(self,image):
		(im_width, im_height) = image.size
		return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

	def run_inference_for_single_image(self,image):
		with self.detection_graph.as_default():
			with tf.Session() as sess:
				# Get handles to input and output tensors
				ops = tf.get_default_graph().get_operations()
				all_tensor_names = {output.name for op in ops for output in op.outputs}
				tensor_dict = {}
				for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
					tensor_name = key + ':0'
					if tensor_name in all_tensor_names:
						tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
				if 'detection_masks' in tensor_dict:
					# The following processing is only for single image
					detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
					detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
					# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
					real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
					detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
					detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
					detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
					detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
					# Follow the convention by adding back the batch dimension
					tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
				image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
				# Run inference
				output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})
				# all outputs are float32 numpy arrays, so convert types as appropriate
				output_dict['num_detections'] = int(output_dict['num_detections'][0])
				output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
				output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
				output_dict['detection_scores'] = output_dict['detection_scores'][0]
				if 'detection_masks' in output_dict:
					output_dict['detection_masks'] = output_dict['detection_masks'][0]
		return output_dict

	def predict(self,image):
		image_ok = self.check_image(image);
		if image_ok is None:
			self.status = 'check image data failed! it is not a image file'
			return self.status
		else:
			self.status = 'check image data success!'
		print(self.status)
		image_np = self.load_image_into_numpy_array(image_ok);
		image_np_expanded = np.expand_dims(image_np, axis=0)
		self.status = 'start to deal iamge and class the object label!'
		output_dict = self.run_inference_for_single_image(image_np)
		self.status = 'class the object label finish! and start to mark label to image data'
		# Visualization of the results of a detection.
		vis_util.visualize_boxes_and_labels_on_image_array(
			image_np,
			output_dict['detection_boxes'],
			output_dict['detection_classes'],
			output_dict['detection_scores'],
			self.category_index,
			instance_masks=output_dict.get('detection_masks'),
			use_normalized_coordinates=True,
			line_thickness=8)
		self.status = 'mark label finish'
		image_final = Image.fromarray(image_np)
		return self.image_to_base64(image_final)

	def predict_ts(self,image):
		image_ok = self.check_image(image);
		if image_ok is None:
			self.status = 'check image data failed! it is not a image file'
			return self.status
		else:
			self.status = 'check image data success!'
		print(self.status)
		image_size = image_ok.size
		image_np = self.load_image_into_numpy_array(image_ok);
		image_np_expanded = np.expand_dims(image_np, axis=0)
		self.status = 'start to deal iamge and class the object label!'
		output_dict = self.run_inference_for_single_image(image_np)
		self.status = 'class the object label finish! and start to mark label to image data'
		# Visualization of the results of a detection.
		return self.get_boxes_and_labels_on_image_array(
				image_size=image_size,
				scores = output_dict['detection_scores'],
				boxes = output_dict['detection_boxes'],
				classes = output_dict['detection_classes'],
				instance_masks=output_dict.get('detection_masks'),
				category_index = self.category_index,
				use_normalized_coordinates=True)
		self.status = 'mark label finish'

	def get_boxes_and_labels_on_image_array(
			self,
			image_size=None,
			min_score_thresh=0.5,
			boxes=None,scores=None,classes=None,
			instance_masks=None,category_index=None,
			groundtruth_box_visualization_color='black',
			agnostic_mode=False,
			skip_scores=False,
			skip_labels=False,
			use_normalized_coordinates=False):
		box_to_display_str_map = collections.defaultdict(list)
		box_to_color_map = collections.defaultdict(str)
		box_to_instance_masks_map = {}
		box_to_instance_boundaries_map = {}
		box_to_keypoints_map = collections.defaultdict(list)
		for i in range(boxes.shape[0]):
			if scores is None or scores[i] > min_score_thresh:
				box = tuple(boxes[i].tolist())
				if instance_masks is not None:
					box_to_instance_masks_map[box] = instance_masks[i]
				if scores is None:
					box_to_color_map[box] = groundtruth_box_visualization_color
				else:
					display_str = ''
					if not skip_labels:
						if not agnostic_mode:
							if classes[i] in category_index.keys():
								class_name = category_index[classes[i]]['name']
							else:
								class_name = 'N/A'
							display_str = str(class_name)
					if not skip_scores:
						if not display_str:
							display_str = '{}%'.format(int(100*scores[i]))
						else:
							display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
					box_to_display_str_map[box].append(display_str)
					if agnostic_mode:
						box_to_color_map[box] = 'DarkOrange'
					else:
						box_to_color_map[box] = groundtruth_box_visualization_color
		result_to_map = {}
		result_to_map['size'] = image_size
		result_to_map['objs'] = list();
		for box, color in box_to_color_map.items():
			ymin, xmin, ymax, xmax = box
			label = box_to_display_str_map[box];
			im_width, im_height = image_size
			if use_normalized_coordinates:
				(left, right, top, bottom) = (xmin * im_width, xmax * im_width,
					ymin * im_height, ymax * im_height)
				result_to_map['objs'].append((label,left, right, top, bottom))
			else:
				result_to_map['objs'].append((label,xmin, xmax, ymin, ymax))
			print("label:{} xmin:{} xmax:{} ymin:{} ymax:{}".format(label,left, right, top, bottom))
		return result_to_map
