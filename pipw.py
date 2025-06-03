## With this script you can install packages using pip and update the requirements.txt file at the same time.
## Usage: python pipw.py install <package_name>

## if used to install a package, no need to rerrun the generate_requirements.py script

import os
import sys

def main():
    # Use pip from the current Python environment
    python = sys.executable
    pip_cmd = f'"{python}" -m pip {" ".join(sys.argv[1:])}'
    os.system(pip_cmd)

    # If an installation is performed, update requirements.txt
    if "install" in sys.argv:
        print("Updating requirements.txt...")
        os.system(f'"{python}" -m pip freeze > requirements.txt')

if __name__ == "__main__":
    main()
