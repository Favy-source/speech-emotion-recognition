"""
Main entry point for the Speech Emotion Recognition system.
"""
import argparse
import asyncio
from pathlib import Path

from src.utils.logger import setup_logger
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition System")
    parser.add_argument("--mode", choices=["train", "api", "realtime"], 
                       required=True, help="Operation mode")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cpu, cuda, auto)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("SER_MAIN")
    logger.info(f"Starting SER system in {args.mode} mode")
    
    # Load configuration
    config = load_config(args.config)
    
    if args.mode == "train":
        from src.training.trainer import train_model
        train_model(config, args.device)
        
    elif args.mode == "api":
        from src.api.main import start_api
        asyncio.run(start_api(config))
        
    elif args.mode == "realtime":
        from src.realtime.processor import start_realtime_processing
        asyncio.run(start_realtime_processing(config))


if __name__ == "__main__":
    main()
