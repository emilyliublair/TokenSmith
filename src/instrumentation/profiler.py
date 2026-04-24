import time
import tracemalloc
import json
from functools import wraps
from pathlib import Path

# A global dictionary to store our execution times and memory usage
profiling_data = {}

def time_it(stage_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            start_time = time.perf_counter()
            
            result = func(*args, **kwargs)
            
            end_time = time.perf_counter()
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            execution_time = end_time - start_time
            peak_mb = peak_mem / (1024 * 1024)
            
            if stage_name not in profiling_data:
                profiling_data[stage_name] = {
                    "total_execution_time_seconds": 0.0,
                    "max_peak_memory_mb": 0.0,
                    "call_count": 0,
                    "calls": [] 
                }
            
            profiling_data[stage_name]["total_execution_time_seconds"] += execution_time
            profiling_data[stage_name]["max_peak_memory_mb"] = max(
                profiling_data[stage_name]["max_peak_memory_mb"], 
                peak_mb
            )
            profiling_data[stage_name]["call_count"] += 1
            
            profiling_data[stage_name]["total_execution_time_seconds"] = round(profiling_data[stage_name]["total_execution_time_seconds"], 4)
            profiling_data[stage_name]["max_peak_memory_mb"] = round(profiling_data[stage_name]["max_peak_memory_mb"], 4)
            
            profiling_data[stage_name]["calls"].append({
                "run": profiling_data[stage_name]["call_count"],
                "execution_time_seconds": round(execution_time, 4),
                "peak_memory_mb": round(peak_mb, 4)
            })
            
            print(f"[{stage_name}] Time: {execution_time:.2f}s | Peak Mem: {peak_mb:.2f} MB")
            
            return result
        return wrapper
    return decorator

def save_profile_report(index_prefix: str = None, output_dir: str = "data"):
    """Saves the aggregated profiling data to a JSON file."""
    
    if index_prefix:
        filename = f"{output_dir}/profiling_report_{index_prefix}.json"
    else:
        filename = f"{output_dir}/profiling_report.json"
        
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(profiling_data, f, indent=4)
    print(f"\n[Profiler] Report saved to {filename}")