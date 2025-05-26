import torch
import os
import sys
import argparse
import time
from model.model import MultiModalAIDetector, classify_file, classify_directory

def main():
    parser = argparse.ArgumentParser(description='AI Content Detector')
    parser.add_argument('--modality', type=str, choices=['speech', 'instrumental', 'mixed_audio', 'image', 'video', 'auto'],
                        default='auto', help='Content modality to process')
    parser.add_argument('--input', type=str, required=True, help='Input file or directory path')
    parser.add_argument('--model', type=str, default='model/model.pth', help='Path to model file')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing (for directory only)')

    args = parser.parse_args()

    # Determine device
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Initialize model
    model = MultiModalAIDetector().to(device)

    # Load model weights if file exists
    if os.path.exists(args.model):
        try:
            model.load_state_dict(torch.load(args.model, map_location=device))
            print(f"Model loaded successfully from {args.model}")
        except Exception as e:
            print(f"Warning: Could not load model from {args.model}: {e}")
            print("Using untrained model. Results will be random.")
    else:
        print(f"Warning: Model file {args.model} not found. Using untrained model. Results will be random.")

    # Process input
    if os.path.isfile(args.input):
        # Process single file
        modality = None if args.modality == 'auto' else args.modality

        if args.benchmark:
            # Run benchmark
            start_time = time.time()

            # Track memory before
            memory_before = None
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated() / 1024 / 1024

            result = classify_file(args.input, model, device)

            # Track memory after
            memory_after = None
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated() / 1024 / 1024

            end_time = time.time()

            print(f"\nResults for {os.path.basename(args.input)}:")
            print(f"Human: {result['Human']:.2f}%")
            print(f"AI: {result['AI']:.2f}%")
            print(f"Processing time: {end_time - start_time:.2f} seconds")

            # Print memory usage
            if memory_before is not None and memory_after is not None:
                print(f"GPU Memory: {memory_after:.2f} MB (change: {memory_after - memory_before:.2f} MB)")

            # Print system memory usage if available
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                print(f"System Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
            except ImportError:
                pass
        else:
            # Normal processing
            result = classify_file(args.input, model, device)

            print(f"\nResults for {os.path.basename(args.input)}:")
            print(f"Human: {result['Human']:.2f}%")
            print(f"AI: {result['AI']:.2f}%")

    elif os.path.isdir(args.input):
        # Process directory
        modality = None if args.modality == 'auto' else args.modality

        if args.benchmark:
            # Run benchmark
            start_time = time.time()

            # Track memory before
            memory_before = None
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated() / 1024 / 1024

            results = classify_directory(args.input, model, device)

            # Track memory after
            memory_after = None
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated() / 1024 / 1024

            end_time = time.time()

            print(f"\nResults for directory {args.input}:")
            for filename, result in results.items():
                if isinstance(result, dict) and 'Human' in result and 'AI' in result:
                    print(f"{filename}: Human: {result['Human']:.2f}%, AI: {result['AI']:.2f}%")
                else:
                    print(f"{filename}: {result}")

            print(f"\nProcessed {len(results)} files in {end_time - start_time:.2f} seconds")
            print(f"Average time per file: {(end_time - start_time) / max(1, len(results)):.2f} seconds")

            # Print memory usage
            if memory_before is not None and memory_after is not None:
                print(f"GPU Memory: {memory_after:.2f} MB (change: {memory_after - memory_before:.2f} MB)")

            # Print system memory usage if available
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                print(f"System Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
            except ImportError:
                pass
        else:
            # Normal processing
            results = classify_directory(args.input, model, device)
    
            print(f"\nResults for directory {args.input}:")
            for filename, result in results.items():
                if isinstance(result, dict) and 'Human' in result and 'AI' in result:
                    print(f"{filename}: Human: {result['Human']:.2f}%, AI: {result['AI']:.2f}%")
                else:
                    print(f"{filename}: {result}")

    else:
        print(f"Error: Input path {args.input} does not exist")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())

"""
Example usage:

python detector.py --input test.wav --modality speech --model model/merged_model.pth 

"""