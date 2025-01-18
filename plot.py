import os
import glob
import re
import matplotlib.pyplot as plt

def extract_losses(logfile):
    """Extract validation and training losses and steps from a log file."""
    train_losses = []
    val_losses = []
    train_steps = []
    val_steps = []
    
    with open(logfile, 'r') as f:
        for line in f:
            # Match lines containing training loss
            train_match = re.search(r'step:(\d+)/\d+\s+train_loss:(\d+\.\d+)', line)
            if train_match:
                step = int(train_match.group(1))
                loss = float(train_match.group(2))
                train_steps.append(step)
                train_losses.append(loss)
            
            # Match lines containing validation loss
            val_match = re.search(r'step:(\d+)/\d+\s+val_loss:(\d+\.\d+)', line)
            if val_match:
                step = int(val_match.group(1))
                loss = float(val_match.group(2))
                val_steps.append(step)
                val_losses.append(loss)
    
    return train_steps, train_losses, val_steps, val_losses

def plot_losses():
    # Find all log files
    log_files = glob.glob('logs/*.txt')
    
    plt.figure(figsize=(12, 6))
    
    # Plot each log file's losses
    for logfile in log_files:
        run_id = os.path.basename(logfile).replace('.txt', '')
        train_steps, train_losses, val_steps, val_losses = extract_losses(logfile)
        
        if train_steps:  # Plot training losses
            plt.plot(train_steps, train_losses, 'b-', alpha=0.3, label=f'Train {run_id}')
        if val_steps:  # Plot validation losses
            plt.plot(val_steps, val_losses, 'r-', marker='o', label=f'Val {run_id}')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.grid(True)
    plt.legend()
    
    # Show the plot instead of saving
    plt.show()

if __name__ == "__main__":
    plot_losses()