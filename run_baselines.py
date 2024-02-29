import subprocess
import json
import os
from datetime import datetime
import re

# Define the path to the baseline configurations directory
config_dir = "baseline_configs"

# Define the name for the log file
log_filename = "algnet_runs.log"


# Function to list all configuration files
def list_configs(config_dir):
    return [f for f in os.listdir(config_dir) if f.endswith(".json")]


# Function for natural sorting
def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


# List and sort configuration files using the natural sort key
config_files = sorted(list_configs(config_dir), key=natural_sort_key)

print("Available configuration files:")
for config_file in config_files:
    print(f"- {config_file}")
print("\nStarting the execution of configurations...\n")

# Open the log file in append mode
with open(log_filename, "a") as log_file:
    # Write the start time of the script execution
    log_file.write(
        f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    log_file.write("Available configuration files:\n")
    for config_file in config_files:
        log_file.write(f"- {config_file}\n")
    log_file.write("\n")

    total_configs = len(config_files)

    # Iterate over each configuration file
    for index, config_file in enumerate(config_files, start=1):
        # Print progress in the terminal
        print(f"Running configuration {index} of {total_configs}: {config_file}")

        # Load and print the configuration in the terminal
        with open(os.path.join(config_dir, config_file), "r") as file:
            config = json.load(file)
            print(json.dumps(config, indent=4))

        # Write the configuration being processed to the log file
        log_file.write(f"\nRunning configuration: {config_file}\n")
        log_file.write(json.dumps(config, indent=4) + "\n")

        # Define the command to run the external script with the current configuration
        command = f"python algnet.py with {os.path.join(config_dir, config_file)}"

        # Run the command and capture the output
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Write the output of the command to the log file
        log_file.write(result.stdout)
        if result.stderr:
            log_file.write("Error:\n" + result.stderr + "\n")

        # Write a separator between configurations in the log for clarity
        log_file.write("\n" + "-" * 50 + "\n")

    # Write the end time of the script execution
    log_file.write(
        f"Script finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
