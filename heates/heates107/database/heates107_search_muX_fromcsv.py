#!/usr/bin/env python 

import argparse
import matplotlib.pyplot as plt

def filter_and_plot_data(lower_energy, upper_energy, filename="XmuDB.csv", plot_flag=False):
    """
    Filters the data based on a specified energy range and optionally plots the results.

    Parameters:
    - low_energy: The lower bound of the energy range.
    - high_energy: The upper bound of the energy range.
    - filename: The name of the CSV file containing the data.
    - plot_flag: Boolean flag to control the plotting of the results.
    """
    atoms = []
    atomic_numbers = []
    initial_states = []
    final_states = []
    energies = []


    # Filter data within the specified energy range
    for line in open(filename):
        line = line.strip().split(",")
        atom, z, initial_state, final_state, energy = line[0], int(float(line[1])), int(float(line[2])), int(float(line[3])), float(line[4])
        
        if lower_energy < energy <= upper_energy:
            atoms.append(atom)
            atomic_numbers.append(z)
            initial_states.append(initial_state)
            final_states.append(final_state)
            energies.append(energy)

            # 整形して出力
            print("Atom: {:<3}, Z: {:<4}, Initial State: {:<5}, Final State: {:<5}, Energy: {:.3f}".format(
                    atom, 
                    z,  
                    initial_state,  
                    final_state,  
                    energy))

    # Plotting if flag is True
    if plot_flag:
        plt.figure(figsize=(10, 6))
        plt.scatter(atomic_numbers, energies, color='blue', label='Energy Levels')
        plt.title('Filtered Atomic Energy Levels')
        plt.xlabel('Atomic Number')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter data based on energy levels and optionally plot the results.")
    parser.add_argument("-l", "--lower_energy", type=float, default=9.7, help="The lower bound of the energy range.")
    parser.add_argument("-u", "--upper_energy", type=float, default=9.9, help="The upper bound of the energy range.")
    parser.add_argument("-f", "--filename", type=str, default="XmuDB.csv", help="The name of the CSV file containing the data.")
    parser.add_argument("-p", "--plot_flag", action="store_true", help="Flag to control the plotting of the results.")

    args = parser.parse_args()

    filter_and_plot_data(args.lower_energy, args.upper_energy, args.filename, args.plot_flag)
