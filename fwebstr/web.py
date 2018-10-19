#!/usr/bin/python

#-*- coding : utf-8 -*-
#
import sys,os
import tornado.ioloop
import tornado.web

base_path = os.path.dirname(__file__);
sys.path.append(os.path.join(base_path,'../'));

from mager import Mager
from index_handler import IndexHandler
from predict_handler import PredictHandler
from predictts_handler import PredictTSHandler
from predict_status import PredictStatusHandler
from upload_handler import UploadHandler

class Application(tornado.web.Application):
	def __init__(self):
		self.mager = Mager();
		self.mager.init();
		handlers = [
			(r"/index",IndexHandler),
			(r"/upload",UploadHandler),
			(r"/predict",PredictHandler,{"mager":self.mager}),
			(r"/game_predict",PredictTSHandler,{"mager":self.mager}),
			(r"/predict_status",PredictStatusHandler,{"mager":self.mager}),
		];
		settings = dict(
				template_path = os.path.join(os.path.dirname(__file__),"templates"),
				static_path = os.path.join(os.path.dirname(__file__),"static"),
				debug = True,
		);
		tornado.web.Application.__init__(self, handlers, **settings);

if __name__=="__main__":

	port = sys.argv[1];
	server = Application();
	server.listen(port);
	tornado.ioloop.IOLoop.instance().start();
