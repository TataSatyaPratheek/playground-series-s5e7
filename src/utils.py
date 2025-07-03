# src/utils.py
import time
import psutil
import logging

def monitor_memory():
    """Monitor memory usage for M1 Air optimization"""
    memory = psutil.virtual_memory()
    return f"Memory: {memory.percent}% used ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)"

def time_function(func):
    """Decorator to time functions"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.2f} seconds")
        return result
    return wrapper
