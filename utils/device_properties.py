import pycuda.driver as cuda
import pycuda.autoinit
import pandas as pd
from tabulate import tabulate

num_devices = cuda.Device.count()
props_all = {}
keys = ['max_registers_per_block', 'max_registers_per_multiprocessor', 'max_shared_memory_per_block',
        'max_memory_per_multiprocessor', 'compute_capability_major', 'compute_capability_minor',
        'concurrent_kernels']

for i in range(num_devices):
    device = cuda.Device(i)
    props = device.get_attributes()
    print(f"Device Name: {device.name()}")
    props_all['Device Name'] = device.name()

    for key, value in props.items():
        if(str(key).lower() in keys):
            print(f"{key}: {value}")
            props_all[str(key)] = value
    
df = pd.DataFrame.from_dict(props_all, orient='index')
print(tabulate(df, headers='keys', tablefmt='pretty'))
# def list_gpu_properties():
#     # Get the number of devices
#     device_count = cuda.Device.count()
    
#     if device_count == 0:
#         print("No GPU devices found.")
#         return
    
#     print(f"{'Device':<10}{'Name':<30}{'Compute Capability':<20}{'Total Memory (MB)':<20}")
#     print("=" * 80)
    
#     for device_id in range(device_count):
#         device = cuda.Device(device_id)
#         device_name = device.name()
#         compute_capability = f"{device.compute_capability()[0]}.{device.compute_capability()[1]}"
#         total_memory = device.total_memory() // (1024 ** 2)  # Convert bytes to MB
        
#         print(f"{device_id:<10}{device_name:<30}{compute_capability:<20}{total_memory:<20}")

# if __name__ == "__main__":
#     list_gpu_properties()

