import sys
from gunicorn.app.wsgiapp import run

if __name__ == '__main__':
    sys.argv = "gunicorn --bind 0.0.0.0:8123 app:app".split()
    sys.exit(run())
