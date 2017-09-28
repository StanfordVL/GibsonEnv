import os
import subprocess

python_path = subprocess.check_output(["which", "python"]).decode("utf-8")
virenv_path = python_path[:python_path.index("/bin")] 
add_on_path = os.path.join(virenv_path, "python3.5", "site-packages")


