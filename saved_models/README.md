Model checkpoints are saved here during training.

Naming convention:
    {config_name}_seed{n}.pt

Examples:
    iq_only_seed0.pt
    full_fusion_seed4.pt

Each .pt file contains:
    - config       : configuration name
    - use_iq/const/spec : active branches
    - seed         : random seed used
    - num_epochs   : total epochs trained
    - best_acc     : best validation accuracy achieved
    - state_dict   : model weights at best epoch
    - train_losses : per-epoch training loss list
    - val_accs     : per-epoch validation accuracy list
    - snr_acc      : per-SNR accuracy dict at best epoch

Note: .pt files are excluded from git tracking by .gitignore due to their size.
All 35 model checkpoints (7 configs × 5 seeds) are available for download:

📦 **[Download Saved Models (Google Drive)](https://drive.google.com/drive/folders/1QkO8FwqrmN--MbtABxjb3MD0byzxk2T8)**
