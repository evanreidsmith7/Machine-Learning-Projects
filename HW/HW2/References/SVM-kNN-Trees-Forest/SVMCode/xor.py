import os
import sys

if "VIRTUAL_ENV" in os.environ:
    print("In a virtual environment (using VIRTUAL_ENV).")
elif sys.prefix != sys.base_prefix:
    print("In a virtual environment (using sys.prefix and sys.base_prefix).")
else:
    print("Not in a virtual environment.")