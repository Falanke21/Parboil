#!/usr/bin/env python3
import subprocess
from statistics import median

if __name__ == "__main__":
	count = 50
	auto_vec_times = []
	
	print("Running")
	for i in range(count):
		result = subprocess.run(
			["./parboil", "run", "mri-gridding", "base", "small"],
			stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
		splited_lst = result.stdout.decode("utf-8").split()
		find_index = splited_lst.index("Compute")
		auto_vec_times.append(float(splited_lst[find_index + 2]))

	median_auto_vec_t = median(auto_vec_times)

	print(f"Compute time: {median_auto_vec_t}s")
