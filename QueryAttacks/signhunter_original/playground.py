import json 
import re
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt 

def asr_under_k_query(nums, k):
    return sum([num < k for num in nums])/len(nums)


log_file = "nes_sign_bandit_standard.log"
f = open(log_file)
lines = f.readlines()

i = 0
attack_name = ""
result_map = defaultdict(list)
while i < len(lines):
    line = lines[i]
    if "config:" in line:
        json_str = line[8:]
        d = eval(json_str)
        attack_name = d['attack_name']
    elif "queries: [" in line:
        while True:
            print(lines[i])
            nums = re.findall(r"\d+\.?\d*e?\+?\d*", lines[i])
            nums = list(map(int, map(float, nums)))
            result_map[attack_name] += nums 
            if "]" in lines[i]:
                break 
            else:
                i += 1
    i += 1

for k, v in result_map.items():
    print(f"{k}: 100: {asr_under_k_query(v, 100)}, 2500: {asr_under_k_query(v, 2500)}")