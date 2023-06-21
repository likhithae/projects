import math
import statistics


class knn():
    def __init__(self):
        self.distances = []

    def calculate_distance(self, x1, x2):
        distance = 0
        for i in range(len(x1)-1):
            distance += (x1[i] - x2[i])**2
        distance = math.sqrt(distance)

        return distance

    def predict_example(self, xTrain, yTrain, xTest, k):
        pred = []
        for x in xTest:
            neighbors_distance = {}
            for idx,data in enumerate(xTrain):
                neighbors_distance[self.calculate_distance(data,x)] = yTrain[idx]
            
            distances = list(set(neighbors_distance.keys()))
            distances.sort()
            distances = distances[:k]
            updated_neighbors = [neighbors_distance[i] for i in distances]
            # print(updated_neighbors)
            pred.append(statistics.mode(updated_neighbors))

        return pred

