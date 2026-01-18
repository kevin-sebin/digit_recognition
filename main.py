import math
import pandas as pd


class KNN:
    def __init__(self, x_train, y_train, k):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
    
    def euclideanDistance(self, x, y):
        distance = 0.0
        for a,b in zip(x, y):
            distance += (a-b)**2
        return distance
    
    def knnModel(self, pred):
        distances = []
        for i, lbl in zip(self.x_train, self.y_train):
            distances.append([self.euclideanDistance(i, pred), lbl])
        distances = sorted(distances, key=lambda x:x[0])
        k_nearest = [int(y) for x,y in distances[:self.k]]
        
        counts = {}
        for lbl in k_nearest:
            if lbl not in counts:
                counts[lbl] = 1
            else:
                counts[lbl] += 1
        pred_lbl = max(counts, key=counts.get)
        return f'Predicted class: {pred_lbl}'



train_df = pd.read_csv('dataset/train.csv')
x_train = (train_df.drop('label', axis=1)/255).values.tolist()
y_train = train_df['label'].values

model = KNN(x_train, y_train, int(math.sqrt(len(x_train))))

