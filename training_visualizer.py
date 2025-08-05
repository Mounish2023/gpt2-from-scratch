import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
import seaborn as sns

def parse_log_file(log_file_path):
    """
    Parse the training log file and extract metrics
    """
    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []
    hella_steps = []
    hella_scores = []
    
    with open(log_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Parse training loss: "Step 1234 | loss: 3.456789"
            train_match = re.search(r'(\d+) train ([\d.]+)', line)
            if train_match:
                step = int(train_match.group(1))
                loss = float(train_match.group(2))
                train_steps.append(step)
                train_losses.append(loss)
            
            # Parse validation loss: "1234 val 3.4567"
            val_match = re.search(r'(\d+) val ([\d.]+)', line)
            if val_match:
                step = int(val_match.group(1))
                loss = float(val_match.group(2))
                val_steps.append(step)
                val_losses.append(loss)
            
            # Parse HellaSwag accuracy: "1234 hella 0.2345"
            hella_match = re.search(r'(\d+) hella ([\d.]+)', line)
            if hella_match:
                step = int(hella_match.group(1))
                score = float(hella_match.group(2))
                hella_steps.append(step)
                hella_scores.append(score)
    
    return {
        'train': {'steps': train_steps, 'losses': train_losses},
        'val': {'steps': val_steps, 'losses': val_losses},
        'hella': {'steps': hella_steps, 'scores': hella_scores}
    }

def create_training_visualizations(data, save_path=None):
    """
    Create comprehensive training visualizations
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Training and Validation Loss over Time
    ax1 = plt.subplot(2, 3, 1)
    
    # Plot training loss (smoothed)
    if len(data['train']['steps']) > 100:
        # Smooth training loss for better visualization
        window_size = len(data['train']['steps']) // 100
        smoothed_train = np.convolve(data['train']['losses'], 
                                   np.ones(window_size)/window_size, mode='valid')
        smoothed_steps = data['train']['steps'][window_size-1:]
        plt.plot(smoothed_steps, smoothed_train, 'b-', alpha=0.8, linewidth=2, 
                label=f'Training Loss (smoothed, window={window_size})')
    
    # Plot raw training loss (with transparency)
    plt.plot(data['train']['steps'], data['train']['losses'], 'b-', 
             alpha=0.3, linewidth=0.5, label='Training Loss (raw)')
    
    # Plot validation loss
    plt.plot(data['val']['steps'], data['val']['losses'], 'r-', 
             linewidth=3, marker='o', markersize=4, label='Validation Loss')
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 2. HellaSwag Accuracy over Time
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(data['hella']['steps'], data['hella']['scores'], 'g-', 
             linewidth=3, marker='s', markersize=6, label='HellaSwag Accuracy')
    plt.xlabel('Training Step')
    plt.ylabel('Accuracy')
    plt.title('HellaSwag Accuracy Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(data['hella']['scores']) * 1.1)
    
    # 3. Loss Improvement Rate
    ax3 = plt.subplot(2, 3, 3)
    if len(data['val']['losses']) > 1:
        val_improvements = []
        val_step_diffs = []
        for i in range(1, len(data['val']['losses'])):
            improvement = data['val']['losses'][i-1] - data['val']['losses'][i]
            step_diff = data['val']['steps'][i] - data['val']['steps'][i-1]
            val_improvements.append(improvement)
            val_step_diffs.append(data['val']['steps'][i])
        
        plt.plot(val_step_diffs, val_improvements, 'purple', 
                linewidth=2, marker='D', markersize=4)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Training Step')
        plt.ylabel('Loss Improvement')
        plt.title('Validation Loss Improvement Rate')
        plt.grid(True, alpha=0.3)
    
    # 4. Training Progress Summary
    ax4 = plt.subplot(2, 3, 4)
    
    # Create summary statistics
    initial_train_loss = data['train']['losses'][0] if data['train']['losses'] else 0
    final_train_loss = data['train']['losses'][-1] if data['train']['losses'] else 0
    initial_val_loss = data['val']['losses'][0] if data['val']['losses'] else 0
    final_val_loss = data['val']['losses'][-1] if data['val']['losses'] else 0
    initial_hella = data['hella']['scores'][0] if data['hella']['scores'] else 0
    final_hella = data['hella']['scores'][-1] if data['hella']['scores'] else 0
    
    categories = ['Initial\nTrain Loss', 'Final\nTrain Loss', 
                 'Initial\nVal Loss', 'Final\nVal Loss',
                 'Initial\nHellaSwag', 'Final\nHellaSwag']
    values = [initial_train_loss, final_train_loss, 
             initial_val_loss, final_val_loss,
             initial_hella, final_hella]
    colors = ['lightblue', 'blue', 'lightcoral', 'red', 'lightgreen', 'green']
    
    bars = plt.bar(categories, values, color=colors, alpha=0.7)
    plt.title('Training Progress Summary')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Loss Distribution
    ax5 = plt.subplot(2, 3, 5)
    
    # Create histograms for loss distributions
    if len(data['train']['losses']) > 10:
        plt.hist(data['train']['losses'], bins=50, alpha=0.7, 
                label='Training Loss', color='blue', density=True)
    if len(data['val']['losses']) > 2:
        plt.hist(data['val']['losses'], bins=20, alpha=0.7, 
                label='Validation Loss', color='red', density=True)
    
    plt.xlabel('Loss Value')
    plt.ylabel('Density')
    plt.title('Loss Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Performance Metrics Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Calculate metrics
    total_steps = max(data['train']['steps']) if data['train']['steps'] else 0
    train_loss_reduction = initial_train_loss - final_train_loss
    val_loss_reduction = initial_val_loss - final_val_loss
    hella_improvement = final_hella - initial_hella
    
    metrics_data = [
        ['Metric', 'Value'],
        ['Total Training Steps', f'{total_steps:,}'],
        ['Initial Training Loss', f'{initial_train_loss:.4f}'],
        ['Final Training Loss', f'{final_train_loss:.4f}'],
        ['Training Loss Reduction', f'{train_loss_reduction:.4f}'],
        ['Initial Validation Loss', f'{initial_val_loss:.4f}'],
        ['Final Validation Loss', f'{final_val_loss:.4f}'],
        ['Validation Loss Reduction', f'{val_loss_reduction:.4f}'],
        ['Initial HellaSwag Accuracy', f'{initial_hella:.4f}'],
        ['Final HellaSwag Accuracy', f'{final_hella:.4f}'],
        ['HellaSwag Improvement', f'{hella_improvement:.4f}'],
    ]
    
    table = ax6.table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                     cellLoc='center', loc='center',
                     colColours=['lightgray', 'lightgray'])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    ax6.set_title('Training Metrics Summary', pad=20, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    return fig

def create_detailed_loss_plot(data, save_path=None):
    """
    Create a detailed loss plot focusing on the training curve
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Top plot: Full training curve
    ax1.plot(data['train']['steps'], data['train']['losses'], 'b-', 
             alpha=0.6, linewidth=1, label='Training Loss')
    ax1.plot(data['val']['steps'], data['val']['losses'], 'r-', 
             linewidth=3, marker='o', markersize=5, label='Validation Loss')
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Complete Training Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Bottom plot: Zoomed in on later training (last 25% of training)
    if len(data['train']['steps']) > 100:
        cutoff_idx = len(data['train']['steps']) * 3 // 4
        late_train_steps = data['train']['steps'][cutoff_idx:]
        late_train_losses = data['train']['losses'][cutoff_idx:]
        
        # Filter validation data for the same period
        late_val_steps = [s for s in data['val']['steps'] if s >= late_train_steps[0]]
        late_val_losses = [data['val']['losses'][i] for i, s in enumerate(data['val']['steps']) if s >= late_train_steps[0]]
        
        ax2.plot(late_train_steps, late_train_losses, 'b-', 
                alpha=0.8, linewidth=1.5, label='Training Loss (Late Training)')
        if late_val_steps:
            ax2.plot(late_val_steps, late_val_losses, 'r-', 
                    linewidth=3, marker='o', markersize=5, label='Validation Loss (Late Training)')
        
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Late Training Phase (Last 25% of Training)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed loss plot saved to {save_path}")
    
    plt.show()
    
    return fig

# Example usage
if __name__ == "__main__":
    # Parse the log file
    log_file_path = "log.txt"  # Update this path to your log file
    
    try:
        data = parse_log_file(log_file_path)
        
        print(f"Parsed data:")
        print(f"Training steps: {len(data['train']['steps'])}")
        print(f"Validation points: {len(data['val']['steps'])}")
        print(f"HellaSwag evaluations: {len(data['hella']['steps'])}")
        
        # Create comprehensive visualizations
        create_training_visualizations(data, save_path="training_progress.png")
        
        # Create detailed loss plot
        create_detailed_loss_plot(data, save_path="detailed_loss_curve.png")
        
        print("\nTraining Summary:")
        if data['train']['losses']:
            print(f"Initial training loss: {data['train']['losses'][0]:.4f}")
            print(f"Final training loss: {data['train']['losses'][-1]:.4f}")
            print(f"Training loss reduction: {data['train']['losses'][0] - data['train']['losses'][-1]:.4f}")
        
        if data['val']['losses']:
            print(f"Initial validation loss: {data['val']['losses'][0]:.4f}")
            print(f"Final validation loss: {data['val']['losses'][-1]:.4f}")
            print(f"Validation loss reduction: {data['val']['losses'][0] - data['val']['losses'][-1]:.4f}")
        
        if data['hella']['scores']:
            print(f"Initial HellaSwag accuracy: {data['hella']['scores'][0]:.4f}")
            print(f"Final HellaSwag accuracy: {data['hella']['scores'][-1]:.4f}")
            print(f"HellaSwag improvement: {data['hella']['scores'][-1] - data['hella']['scores'][0]:.4f}")
            
    except FileNotFoundError:
        print(f"Log file not found at {log_file_path}")
        print("Please update the log_file_path variable to point to your log file.")
        
        # Create sample data for demonstration
        print("Creating sample visualization with demo data...")
        sample_data = {
            'train': {
                'steps': list(range(0, 1000, 5)),
                'losses': [10.0 * np.exp(-x/200) + 3.0 + 0.1*np.random.randn() for x in range(0, 1000, 5)]
            },
            'val': {
                'steps': list(range(0, 1000, 50)),
                'losses': [10.0 * np.exp(-x/200) + 3.0 for x in range(0, 1000, 50)]
            },
            'hella': {
                'steps': list(range(0, 1000, 50)),
                'scores': [0.25 + 0.05 * (1 - np.exp(-x/300)) for x in range(0, 1000, 50)]
            }
        }
        create_training_visualizations(sample_data)