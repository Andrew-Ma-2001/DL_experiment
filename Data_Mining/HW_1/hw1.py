"""
2021/04/08 MYZ
智能数据挖掘作业 1
"""
import numpy as np

array_1 = [22, 1, 42, 10]
array_2 = [20, 0, 36, 8]

square_sum = 0
abs_sum = 0
max_sum = 0
minikowski_sum = 0
for i in range(0, 3):
    square_sum += np.square(array_1[i] - array_2[i])
    abs_sum += np.abs(array_1[i] - array_2[i])
    max_sum = max(max_sum, np.abs(array_1[i] - array_2[i]))
    minikowski_sum += np.power(array_1[i] - array_2[i], 3)

dis_1 = np.sqrt(square_sum)
dis_2 = abs_sum
dis_3 = np.power(minikowski_sum, 1/3)
dis_4 = max_sum

print(f' 1. 欧几里得距离为{dis_1}\n 2. 曼哈顿距离为{dis_2}\n '
      f'3. q=3 闵可夫斯基距离为{dis_3}\n 4. 上确界距离为{dis_4}')
