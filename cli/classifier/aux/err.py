"""Handle error information and exit."""


import sys


def err(msg):
    """Write a message to stderr."""
    sys.stderr.write("ERROR: {}\n".format(msg))

def err_exit(msg=None, traceback=None, parser=None, exit_code=1):
    """Give error information and exit with a given exit code (1 by default)."""
    if msg:
        err(msg)
    if traceback:
        sys.stderr.write(traceback)
    if parser:
        parser.print_help()
    sys.exit(exit_code)
