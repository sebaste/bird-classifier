"""Bird classifier CLI application."""

#!/usr/bin/env python3


import logging
import os
import sys
import time
import traceback
import yappi
if sys.version_info < (3, 0):
    sys.stderr.write("ERROR: Python 2.x is not supported - Python >= 3.0 required\n")
    sys.exit(1)

from init import init
from classification.BirdClassifier import classify_birds
from classification.classification import Classification, Tf
from aux.err import err_exit
from aux.timec import timec
from aux.url_open import url_open


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _get_args(config, _time, _profile):
    args = None
    try:
        args = {
                'nm_top_results': int(config.get('classifier', 'nm_top_results')),
                'multiprocessing_threshold': int(config.get('classifier',
                    'multiprocessing_threshold')),
                'tfhub_cache_dir': config.get('classifier', 'tfhub_cache_dir'),
                'url_model': config.get('bird-classifier', 'url_model'),
                'url_labels': config.get('bird-classifier', 'url_labels'),
                }
    except configparser.NoOptionError as e:
        err_msg = "Invalid config file"
        logging.exception(err_msg)
        raise Exception("%s: %s" % (err_msg, e))
    args['time'] = _time
    args['profile'] = _profile
    return args

def main():
    try:
        with timec() as t:
            config, data, _time, _profile, show = init()

            args = _get_args(config, _time, _profile)

            if _profile:
                yappi.set_clock_type("cpu")
                yappi.start()

            Tf.init(args['tfhub_cache_dir'])
            results = classify_birds(data, config, args)

            if _profile:
                yappi.get_func_stats().print_all()
                yappi.get_thread_stats().print_all()

            if show:
                import timg
                from PIL import Image
                import requests
                obj = timg.Renderer()
                for res in results:
                    img = Image.open(requests.get(res.image, stream=True).raw)
                    obj.load_image(img)
                    obj.resize(120,40)
                    obj.render(timg.Ansi8HblockMethod)
                    print(str(res))
            else:
                for res in results:
                    print(str(res))
        if _time:
            logging.debug(f"Time taken for application: {t():.4f}s")
    except Exception as e:
        logging.exception("Unexpected error - exiting")
        err_exit(msg=e, traceback=traceback.format_exc(), exit_code=1)

if __name__ == "__main__":
    main()
