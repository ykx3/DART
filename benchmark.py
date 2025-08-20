"""
Model Performance Benchmarking Tool

This module provides comprehensive benchmarking capabilities for deep learning models,
specifically designed for testing model performance across different batch sizes.
It measures latency and throughput to help optimize model deployment.

Key Features:
- Automated batch size testing with exponential growth
- Memory-aware testing with OOM handling
- Comprehensive performance metrics (latency, throughput)
- CSV output for result analysis
- Support for various model architectures via timm

Usage:
    python benchmark.py --model deit_small_patch16_224 --max-batch-size 1024
    python benchmark.py --model resnet50 --device cpu --iterations 50
"""

import torch
import timm
import time
import numpy as np
import pandas as pd
import argparse
import models_deit
import models_mamba

# ==============================================================================
# 1. Command Line Argument Parser
# ==============================================================================

def parse_arguments():
    """
    Parse command line arguments for the benchmarking tool.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Model Performance Benchmarking Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        '--model', 
        type=str, 
        default='deit_small_patch16_224',
        help='Model name to benchmark (timm model name)'
    )
    parser.add_argument(
        '--img-size', 
        type=int, 
        default=224,
        help='Input image size'
    )
    parser.add_argument(
        '--num-classes', 
        type=int, 
        default=1000,
        help='Number of output classes'
    )
    
    # Device configuration
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to run on (auto: automatically detect CUDA)'
    )
    parser.add_argument(
        '--dtype', 
        type=str, 
        default='float32',
        choices=['float32', 'float16', 'bfloat16'],
        help='Data type for tensors'
    )
    
    # Benchmark configuration
    parser.add_argument(
        '--max-batch-size', 
        type=int, 
        default=1024,
        help='Maximum batch size to test (will test powers of 2 up to this value)'
    )
    parser.add_argument(
        '--warmup', 
        type=int, 
        default=20,
        help='Number of warmup iterations'
    )
    parser.add_argument(
        '--iterations', 
        type=int, 
        default=100,
        help='Number of timing iterations'
    )
    
    # Output configuration
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Output CSV filename (default: benchmark_{model}.csv)'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()

# ==============================================================================
# 2. Core Benchmarking Function
# ==============================================================================

def run_benchmark(model, batch_size, device, dtype, img_size, num_warmup, num_iterations, verbose=False):
    """
    Execute complete warmup and benchmark testing for a given batch size.
    
    This function performs comprehensive performance testing including warmup runs
    and timing measurements. It handles out-of-memory errors gracefully and
    returns detailed performance metrics.
    
    Args:
        model (torch.nn.Module): The model to benchmark
        batch_size (int): Batch size to test
        device (str): Device to run on ('cuda' or 'cpu')
        dtype (torch.dtype): Data type for tensors
        img_size (int): Input image size
        num_warmup (int): Number of warmup iterations
        num_iterations (int): Number of timing iterations
        verbose (bool): Enable verbose output
    
    Returns:
        dict or None: Dictionary containing benchmark results if successful,
                     None if OOM or other error occurs
    
    Example:
        >>> model = timm.create_model('deit_small_patch16_224')
        >>> result = run_benchmark(model, 32, 'cuda', torch.float32, 224, 20, 100)
        >>> print(result)  # {'batch_size': 32, 'latency_ms': 15.2, 'throughput_fps': 2105.3}
    """
    if verbose:
        print(f"\n--- Starting test for Batch Size: {batch_size} ---")
    
    try:
        # 1. Create input tensor
        input_shape = (batch_size, 3, img_size, img_size)
        dummy_input = torch.randn(input_shape, device=device, dtype=dtype)
        
        # 2. Warmup phase
        if verbose:
            print("  > Warming up...", end='')
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)
                if device == 'cuda':
                    torch.cuda.synchronize()
        if verbose:
            print("Done.")

        # 3. Timing test
        if verbose:
            print("  > Running timing test...", end='')
        timings = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                _ = model(dummy_input)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                timings.append(end_time - start_time)
        if verbose:
            print("Done.")

        # 4. Calculate results
        total_time_seconds = np.sum(timings)
        avg_latency_ms = np.mean(timings) * 1000
        throughput_fps = (num_iterations * batch_size) / total_time_seconds

        if verbose:
            print(f"  > Results: Avg latency={avg_latency_ms:.2f} ms/batch, Throughput={throughput_fps:.2f} samples/sec")
        
        # Clean up memory
        del dummy_input
        if device == 'cuda':
            torch.cuda.empty_cache()

        return {
            'batch_size': batch_size,
            'latency_ms': avg_latency_ms,
            'throughput_fps': throughput_fps
        }

    except torch.cuda.OutOfMemoryError:
        if verbose:
            print("\n  > Failed: CUDA Out of Memory (insufficient memory)")
            print(f"--- Batch Size: {batch_size} test failed ---")
        if device == 'cuda':
            torch.cuda.empty_cache()
        return None  # Return None to indicate test failure
        
    except Exception as e:
        if verbose:
            print(f"\n  > Failed: Unknown error occurred: {e}")
        return None

# ==============================================================================
# 3. Main Execution Flow
# ==============================================================================

def main():
    """
    Main execution function for the benchmarking process.
    
    This function orchestrates the entire benchmarking workflow:
    1. Parses command line arguments
    2. Loads the specified model
    3. Tests batch sizes in exponential progression
    4. Handles errors and memory issues
    5. Saves results to CSV file
    
    Returns:
        None
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Set dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    # Set output filename
    if args.output is None:
        output_filename = f"benchmark_{args.model}.csv"
    else:
        output_filename = args.output
    
    print("="*50)
    print("Model Performance Benchmarking Tool")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Data Type: {args.dtype}")
    print(f"Image Size: {args.img_size}")
    print(f"Max Batch Size: {args.max_batch_size}")
    print(f"Warmup Iterations: {args.warmup}")
    print(f"Timing Iterations: {args.iterations}")
    print(f"Output File: {output_filename}")
    print("="*50)

    # Prepare model
    try:
        model = timm.create_model(
            args.model, 
            pretrained=False, 
            img_size=args.img_size, 
            num_classes=args.num_classes
        )
        model.eval()
        model.to(device=device, dtype=dtype)
        if args.verbose:
            print(f"Model loaded successfully: {args.model}")
    except Exception as e:
        print(f"Error loading model {args.model}: {e}")
        return

    all_results = []
    current_batch_size = 1

    # Test batch sizes in powers of 2
    while current_batch_size <= args.max_batch_size:
        # Run benchmark for single batch size
        result = run_benchmark(
            model=model,
            batch_size=current_batch_size,
            device=device,
            dtype=dtype,
            img_size=args.img_size,
            num_warmup=args.warmup,
            num_iterations=args.iterations,
            verbose=args.verbose
        )

        if result:
            all_results.append(result)
            if not args.verbose:
                print(f"Batch {current_batch_size}: {result['latency_ms']:.2f}ms, {result['throughput_fps']:.1f} fps")
        else:
            # If OOM or any error occurs, terminate testing
            if args.verbose:
                print("\nTesting terminated early due to errors.")
            break
        
        # Update to next power of 2
        current_batch_size *= 2

    # ==============================================================================
    # 4. Print final summary results
    # ==============================================================================
    if not all_results:
        print("\nFailed to successfully complete any batch size tests.")
        return

    print("\n\n" + "="*50)
    print("           Final Benchmark Summary Results")
    print("="*50)
    
    # Use pandas for formatted output, more aesthetically pleasing
    df = pd.DataFrame(all_results)
    df = df.set_index('batch_size')
    print(df.to_string(float_format="%.2f"))

    print("="*50)

    try:
        df.to_csv(output_filename)
        print(f"\nResults successfully saved to file: {output_filename}")
    except Exception as e:
        print(f"\nError saving file: {e}")
    # ------------------------------------

if __name__ == '__main__':
    main()