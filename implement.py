import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm
from projectstatistics import ProjectStatistics
import matplotlib.pyplot as plt
import os 

current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'Khảo_sát_việc_đọc_sách.csv')

model = ProjectStatistics()
data = model.load_data(file_path=file_path)
read_time, read_like = model.process_data()
confidence = float(input('inputing the confidence score: '))
mean, lower_bound_time, upper_bound_time = model.estimation_point_mean(confidence=confidence)
p_sample, lower_bound_like, upper_bound_like = model.estimation_proportion_likereadingbooks(
    confidence=confidence)

print(f"The range of average time for reading books of Hanoi students(confidence {confidence*100}%): ({
      lower_bound_time},{upper_bound_time})")
print(f"{mean} +- {mean - lower_bound_time}")
print(f"The range of proportion that Hanoi students like reading books(confidence {confidence*100}%): ({
      lower_bound_like},{upper_bound_like})")
print(f"{p_sample} +- {p_sample - lower_bound_like}")


# EDA
plt.boxplot(read_time, vert=False)
# plt.xlim(0, 25)
plt.xlabel('reading time')
plt.show()

like = p_sample
dislike = 1 - p_sample
proportions = [like, dislike]
plt.pie(proportions)
plt.legend(['like', 'dislike'])
plt.show()
