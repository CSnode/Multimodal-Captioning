# -*- coding: utf-8 -*-
import httplib
import json
c = httplib.HTTPConnection('localhost', 8080)
req_obj = {'input': '/tmp/demo/upload/0a2ff70d494954a8158be13f2197baa398686413.jpg',
		   'output': '/tmp/demo/result',
		   'threshold': 0.5}
c.request('POST', '/process', json.dumps(req_obj))
doc = json.loads(c.getresponse().read())
print doc

