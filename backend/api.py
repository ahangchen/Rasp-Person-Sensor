def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return _singleton


@singleton
class ApiConfig(object):
    host = 'http://222.201.137.47:12346'
    urls = {
        'upload_detect_info': host + '/sensor/vision',
        'upload_img': host + '/file/img',
    }
    def __init__(self, x=0):
        self.x = x  