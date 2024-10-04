import subprocess

# Path to your requirements file
requirements_file = "requirements.txt"

# Open the file and install each package one by one
with open(requirements_file, 'r') as file:
    for line in file:
        package = line.strip()
        if package:
            print(f"Installing {package}...")
            subprocess.run(["pip", "install", package])
