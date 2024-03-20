import sys
import subprocess

subprocess.check_call([sys.argv[0], './api.py'])
subprocess.check_call([sys.argv[0], './full.py'])


