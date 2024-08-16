"""Simple wrapper for urllib.request.urlopen to make it retry on non-fatal failures."""

import logging
import time
import ssl
import urllib.request


class UrlOpenFatalException(Exception):
    """Exception signifying an error no reason to retry for."""


def url_open(url):
    """Call urllib.request.urlopen() and retry on non-fatal failures."""

    response = None
    retry = 1
    retry_max = 5
    retry_wait = 1

    while True:
        rerun = False

        try:
            response = urllib.request.urlopen(url)
        except (ValueError, ssl.SSLError, urllib.error.HTTPError, urllib.error.URLError) as err:
            logging.warning("Error opening URL '%s': %s", url, err)
            raise UrlOpenFatalException
        except Exception:
            if retry == retry_max:
                logging.warning("Opening URL '%s' failed after %d retries", url, retry_max)
                raise UrlOpenFatalException
            logging.debug("Error opening URL '%s' - making retry %d/%d", url, retry, retry_max,
                    exc_info=True)
            retry += 1
            rerun = True
            time.sleep(retry_wait)

        if not rerun:
            break

    return response
