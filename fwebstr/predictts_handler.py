#!/usr/bin/python
#-*- coding : utf-8 -*-

import sys,os
#=============================================
''' import common module '''
base_path = os.path.dirname(__file__);
sys.path.append(os.path.join(base_path,'../commons'));
#=============================================

import tornado.web
from logger import *
import common
from handler import RequestHandler

class PredictTSHandler(RequestHandler):

	def __init__(self,*args, **kwargs):
		RequestHandler.__init__(self, *args, **kwargs);

	def initialize(self,mager):
		self.mager = mager;

	@tornado.gen.coroutine
	@common.json_loads_body
	def post(self):
		try:
			if not 'data' in self.body_json:
				self.except_handle('the url data format error');
				return ;
			idata = self.body_json['data'];
			print('get data file success')
			if len(idata) == 0:
				self.except_handle('the param text is empty');
				return ;
			rest = self.mager.predict_ts(idata);
			self.write(self.gen_result(0,'enjoy success',rest));
		except Exception as e:
			logging.error(str(e));
			raise e;
