#!/usr/bin/python
#-*- coding:utf-8 -*-
import os,sys,json

#============================================
''' import MyException module '''
base_path = os.path.dirname(__file__);
sys.path.append(base_path);

#============================================
from logger import *

def print_dic(struct):
	value = json.dumps(struct,indent = 4,ensure_ascii=False);
	print(value);

def get_dicstr(struct):
	value = json.dumps(struct,indent = 4,ensure_ascii=False);
	return value;

def singleton(cls,*args,**kw):
	instances = {};
	def __singleton():
		if cls not in instances:
			instances[cls] = cls(*args,**kw);
		return instances[cls];
	return __singleton;

def json_loads_body(func):
	def wrapper(self, *args, **kwargs):
		try:
			if not self.request.body is None:
				self.body_json = json.loads(self.request.body.decode('utf-8'));
		except Exception as e:
			raise e;
		return func(self, *args, **kwargs);
	return wrapper;
