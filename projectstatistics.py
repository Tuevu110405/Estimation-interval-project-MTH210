import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm




class ProjectStatistics:
    def __init__(self):
        self.file_path = None
        self.data = None
        self.reading_time = None
        self.reading_like = None


    def load_data(self, file_path):
        self.file_path = file_path
        try:
            self.data = pd.read_csv(self.file_path) 
            

            #cleaning time data
            self.data["Thời gian bạn đọc sách trung bình trong một ngày"] = (
    self.data["Thời gian bạn đọc sách trung bình trong một ngày"].str.strip() + ":00"
)
            
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            raise  # Re-raise the exception to halt execution
        

    def process_data(self):
        self.reading_time = pd.to_timedelta(self.data.iloc[:, 3]).dt.total_seconds() / 3600
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.reading_like = encoder.fit_transform(self.data.iloc[:, 2].values.reshape(-1, 1))
        return self.reading_time, self.reading_like

    def estimation_point_mean(self, confidence):
        #Compute mean and stadard deviation
        mean = np.mean(self.reading_time)
        std = np.std(self.reading_time, ddof=1)
        n = len(self.reading_time)
        unbiased_std = std * np.sqrt(n / (n - 1))
        #Getting z value with scipy
        z_score = norm.ppf(1 - (1 - confidence) / 2)
        #compute margin of error
        margin_of_error = z_score * (unbiased_std / np.sqrt(n))
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        return mean, lower_bound, upper_bound

    def no_of_data(self):
        return len(self.reading_time)

    def estimation_proportion_likereadingbooks(self, confidence):
        n = len(self.reading_like)
        num_like = 0
        for i in self.reading_like:
            num_like += i[0]

        #Compute rate of liking reading books
        p_sample = num_like / n
        #Getting z value
        z_score = norm.ppf(1 - (1 - confidence) / 2)
        #compute p margin of errors
        margin_of_error = z_score * np.sqrt((p_sample * (1 - p_sample)) / n)
        lower_bound = p_sample - margin_of_error
        upper_bound = p_sample + margin_of_error
        return p_sample, lower_bound, upper_bound

        




