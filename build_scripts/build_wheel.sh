# vi /var/tmp/test_script.sh
#!/bin/bash

# Install pip requirements
pip install wheel auditwheel twine
pip install -r build_scripts/requirements.txt
pip install -e .
python$P_VERSION setup.py bdist_wheel
auditwheel repair dist/*

