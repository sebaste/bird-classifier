import argparse
import configparser
import logging
import multiprocessing
import os
import re
import stat
import sys
import traceback

from aux.err import *
from aux.MultiprocessingLog import MultiprocessingLog


def _get_config():
    CONFIG_FILE = "config.ini"
    if not os.path.exists(CONFIG_FILE):
        raise Exception("Config file '{}' does not exist".format(CONFIG_FILE))
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config

def init():
    config = _get_config()

    parser = argparse.ArgumentParser(description='Classifier')

    parser.add_argument('files', nargs='*', type=str,
            help='The file(s) containing image URLs to open. Opening more than one file leads to a concatenation of all files.')
    parser.add_argument('-i', '--image', nargs=1, type=str, action='append',
            help='Image URL.') 
    parser.add_argument('-d', '--debug', action='store_true',
            help='Turn on debug output.')
    parser.add_argument('-t', '--time', action='store_true',
            help='Time the execution time(s) of the application.')
    parser.add_argument('-p', '--profile', action='store_true',
            help='Profile the application')
    parser.add_argument('-s', '--show', action='store_true',
            help='Show image in terminal window')

    args = parser.parse_args()

    mpl = MultiprocessingLog("classifier.log", mode="w+", maxsize=0, rotate=0)
    formatter = logging.Formatter("[%(asctime)s] %(processName)s: %(levelname)s: %(message)s")
    mpl.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(mpl)
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.debug("Python version used: {}".format(sys.version_info))
        logging.debug("args: {}".format(args))
    else:
        logger.setLevel(logging.INFO)

    # Check if stdin is piped and if it is redirected.
    mode = os.fstat(0).st_mode
    stdin_is_piped, stdin_is_redirected = stat.S_ISFIFO(mode), stat.S_ISREG(mode)
    logging.debug("stdin_is_piped={}, stdin_is_redirected={}".format(stdin_is_piped, stdin_is_redirected))
    sys_stdin = stdin_is_piped or stdin_is_redirected
    # Check if stdout is redirected.
    stdout_is_redirected = not sys.stdout.isatty()
    logging.debug("stdout_is_redirected={}".format(stdout_is_redirected))

    e = []
    if not args.files and not sys_stdin:
        e += ["file(s)"]
    if not args.image:
        e += ["image(s)"]
    if len(e) == 2:
        err_exit(msg="no {} and no {} provided as argument(s)\n".format(e[0], e[1]), parser=parser)
    del e

    data = []

    # Get any standard input directed to this program.
    if sys_stdin:
        data = [line.strip() for line in sys.stdin]
    if data:
        logging.debug("Read {} line(s) from sys.stdin".format(len(data)))

    # If more than one file is given, concatenate them together in the sequence that they arrived.
    for f in args.files:
        f_data_prev_len = len(data)
        try:
            with open(f, 'r') as rf:
                data += [line.strip() for line in rf]
        except FileNotFoundError as e:
            err_exit(traceback=traceback.format_exc())
        logging.debug("Read " + str(len(data) - f_data_prev_len) + " line(s) from file " + str(f))
    f_data_nm_lines = len(data)

    # Add any number of image URL(s) stated with the flag.
    if args.image:
        for image in args.image:
            data += image

    return config, data, args.time, args.profile, args.show
