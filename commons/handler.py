#!/usr/bin/python
#-*- coding:utf-8 -*-
import json,traceback
import tornado.web
from logger import *
#==================================================

class RequestHandler(tornado.web.RequestHandler):

	def __init__(self,*args,**kwargs):
		tornado.web.RequestHandler.__init__(self, *args, **kwargs);

	def set_default_headers(self):
		self.set_header('Access-Control-Allow-Origin', '*')
		self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
		self.set_header('Access-Control-Max-Age', 1000)
		self.set_header('Access-Control-Allow-Headers', '*')
		self.set_header('Content-type', 'application/json')

	def write(self, trunk):
		if type(trunk) == int:
			trunk = str(trunk);
		super(RequestHandler, self).write(trunk)

	def gen_result(self, code, message, result):
		dic = dict();
		dic['code'] = code;
		dic['message'] = message
		if result is None:
			return json.dumps(dic, sort_keys=False,ensure_ascii=False);
		dic['result'] = result;
		return json.dumps(dic, sort_keys=False,ensure_ascii=False);

	def except_handle(self, message):
		s = traceback.format_exc();
		logging.error(s + message);
		msg = message.replace(',',' ').replace('\n','#');
		msg = msg.replace('"',' ');
		msg = msg.replace(';',' ');
		self.write(self.gen_result(-1,msg, None))
		return
