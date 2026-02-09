import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate

# Set up plot parameters to use Agg backend and save plots directly without showing them.
matplotlib.use("Agg")

def generate_sine_wave_data(frequency, duration):
    """Generate sine wave data using NumPy."""
    t = np.linspace(0, duration, int(duration * 100), endpoint=False)
    y = np.sin(2 * np.pi * frequency * t)
    return t, y

def plot_and_save(data, title, xlabel, ylabel, filename):
    """Plot the given data and save it to a file."""
    plt.plot(data[0], data[1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    try:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"SAVED: {filename}")
    except Exception as e:
        print(f"Error saving plot to file: {e}")

def main():
    # Generate sine wave data
    frequency = 1.0  # Frequency in Hz
    duration = 2     # Duration of the signal in seconds

    t, y = generate_sine_wave_data(frequency, duration)

    print("Generated Sine Wave Data:")
    print(t)
    print(y)

    filename = "sine_wave.png"
    
    plot_and_save((t, y), 'Sine Wave', 'Time (seconds)', 'Amplitude', filename)

if __name__ == "__main__":
    main()