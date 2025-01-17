import os
import glob
import re
import matplotlib.pyplot as plt

def extract_val_losses(logfile):
    """Extract validation losses and steps from a log file."""
    losses = []
    steps = []
    
    with open(logfile, 'r') as f:
        for line in f:
            # Match lines containing validation loss
            match = re.search(r'step:(\d+)/\d+\s+val_loss:(\d+\.\d+)', line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                steps.append(step)
                losses.append(loss)
    
    return steps, losses

def plot_validation_losses():
    # Find all log files
    log_files = glob.glob('logs/*.txt')
    
    plt.figure(figsize=(12, 6))
    
    # Plot each log file's validation losses
    for logfile in log_files:
        run_id = os.path.basename(logfile).replace('.txt', '')
        steps, losses = extract_val_losses(logfile)
        
        if steps:  # Only plot if we found validation losses
            plt.plot(steps, losses, marker='o', label=f'Run {run_id}')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Validation Loss')
    plt.title('Training Validation Losses Across Different Runs')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig('validation_losses_comparison.png')
    print(f"Plot saved as validation_losses_comparison.png")

if __name__ == "__main__":
    plot_validation_losses()