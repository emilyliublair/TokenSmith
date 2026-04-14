import time
import functools
import json
from pathlib import Path

# A global dictionary to store our execution times
profiling_data = {}

def time_it(stage_name):
    """
    A decorator to measure the execution time of a pipeline stage.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            # Execute the actual function
            result = func(*args, **kwargs)
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Log the time
            if stage_name not in profiling_data:
                profiling_data[stage_name] = []
            profiling_data[stage_name].append(execution_time)
            
            print(f"[Profiler] {stage_name} took {execution_time:.4f} seconds")
            return result
        return wrapper
    return decorator

def save_profile_report(filename="data/profiling_report.json"):
    """Saves the aggregated profiling data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(profiling_data, f, indent=4)
    print(f"\n[Profiler] Report saved to {filename}")