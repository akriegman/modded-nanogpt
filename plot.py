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
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot training losses
    for logfile in log_files:
        run_id = os.path.basename(logfile).replace('.txt', '')
        train_steps, train_losses, _, _ = extract_losses(logfile)
        
        if train_steps:
            ax1.plot(train_steps, train_losses, 'b-', alpha=0.3, label=f'Train {run_id}')
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim(0, 4.3)
    
    # Plot validation losses
    for logfile in log_files:
        run_id = os.path.basename(logfile).replace('.txt', '')
        _, _, val_steps, val_losses = extract_losses(logfile)
        
        if val_steps:
            ax2.plot(val_steps, val_losses, 'r-', marker='o', label=f'Val {run_id}')
    
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Losses')
    ax2.grid(True)
    ax2.legend()
    ax2.set_ylim(0, 4.3)
    
    # Add some spacing between subplots
    plt.tight_layout()
    
    # Save the plots
    plt.savefig('loss.png')
    print(f"Plot saved as loss.png")

if __name__ == "__main__":
    plot_losses()