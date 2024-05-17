#!/usr/bin/env python

import csv
import subprocess
import os

# Define the filename
filename = 'data.csv'

# Open the file and read the content
with open(filename, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row

    for row in reader:
        if len(row) == 4:
            obsid, name, key, svpath = row

            # Create the directory for the current name
            if not os.path.exists(name):
                os.makedirs(name)
            
            # Change to the created directory
            os.chdir(name)
            
            # Construct the bash command
            bash_command = f'wget -nv -m -np -nH --cut-dirs=6 -R "index.html*" --execute robots=off --wait=1 {svpath}/{obsid}/ .'
            print("CMD: ",bash_command)
            # Execute the bash command
            try:
                result = subprocess.run("pwd; date", shell=True, check=True, capture_output=True, text=True)                
                print(f"Start for {name} ({obsid}): {result.stdout}")
                
                result = subprocess.run(bash_command, shell=True, check=True, capture_output=True, text=True)
                print(f"End   for {name} ({obsid}): {result.stdout}")
                
            except subprocess.CalledProcessError as e:
                print(f"Error for {name} ({obsid}): {e.stderr}")

            # Change back to the original directory
            os.chdir('..')            
