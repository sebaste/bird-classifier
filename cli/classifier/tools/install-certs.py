# Install or update a set of default root certificates for the SSL module.
# Uses the certificates provided by the certifi package (https://pypi.python.org/pypi/certifi).

import os
import os.path
import ssl
import stat
import subprocess
import sys


STAT_0o775 = ( stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
             | stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP
             | stat.S_IROTH |                stat.S_IXOTH )


def main():
    openssl_dir, openssl_cafile = os.path.split(
        ssl.get_default_verify_paths().openssl_cafile)

    subprocess.check_call([sys.executable,
        "-E", "-s", "-m", "pip", "install", "--upgrade", "certifi"])

    import certifi

    os.chdir(openssl_dir)
    relpath_to_certifi_cafile = os.path.relpath(certifi.where())
    try:
        os.remove(openssl_cafile)
    except FileNotFoundError:
        pass
    os.symlink(relpath_to_certifi_cafile, openssl_cafile)
    os.chmod(openssl_cafile, STAT_0o775)

if __name__ == '__main__':
    main()
