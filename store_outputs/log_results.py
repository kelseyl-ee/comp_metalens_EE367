import csv
import os

def log_result(dataset_name, method, num_kernels, kernel_size, accuracy):
    """
    Appends a result row to the specific CSV for the dataset.
    dataset_name should be 'MNIST', 'Fashion', or 'CIFAR_G'
    """
    file_path = f"{dataset_name}_class_results.csv"
    
    # Data row to append
    row = [method, num_kernels, kernel_size, f"{accuracy:.2f}%"]
    
    try:
        with open(file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"Result logged to {file_path}")
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Did you run the Init script?")






