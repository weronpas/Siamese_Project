import torch
import yaml
import argparse

from src.train import run_training


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Siamese Network Orchestrator")
    parser.add_argument('--config', type=str, default='configs/baseline.yaml', 
                        help='Path to the YAML configuration file')
    parser.add_argument('--mode', type=str, default='train',
                        help='Execution mode: train')
    args = parser.parse_args()


    config = load_config(args.config)

    print("=========================================")
    print("      SIAMESE NETWORK ORCHESTRATOR       ")
    print("=========================================")
    print(f"Active Mode:    {args.mode.upper()}")
    print(f"Configuration: {args.config}")
    print("=========================================\n")
    
    device = torch.device("mps")
    
    if args.mode == 'train':
        run_training(
            data_dir=config['paths']['data_dir'],
            batch_size=config['hyperparameters']['batch_size'],
            epochs=config['training']['epochs'],
            lr=config['hyperparameters']['learning_rate'],
            margin=config['hyperparameters']['margin'],
            patience=config['training']['patience'],
            device=device
        )

if __name__ == "__main__":
    main()