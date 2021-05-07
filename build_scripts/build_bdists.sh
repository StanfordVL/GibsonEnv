# vi /var/tmp/test_script.sh
#!/bin/bash

# Install pip requirements
export CC=/usr/lib64/openmpi/bin/mpicc

for version in "${!python_versions[@]}"; do
  pip$version install wheel auditwheel twine
  pip$version install -r build_scripts/requirements.txt
  pip$version install -e .
  python$version setup.py bdist_wheel
done

auditwheel repair dist/*.whl

