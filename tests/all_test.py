import os
import sys
import subprocess

d = os.path.dirname(__file__)

# we can only register -> build once per process so need to spawn multiple python processes

def test_rand():
    subprocess.check_call([sys.executable, os.path.join(d, 'rand.py')])

def test_api():
    subprocess.check_call([sys.executable, os.path.join(d, 'api.py')])

def test_full():
    subprocess.check_call([sys.executable, os.path.join(d, 'full.py')])

def test_multi():
    subprocess.check_call([sys.executable, os.path.join(d, 'multi.py')])


