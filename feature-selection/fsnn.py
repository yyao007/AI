import time
import numpy
import Queue

class KNN_classifier:
    def Euclidean_distance(self, p1, p2, feature_set):
        distance = 0.0;
        for i in feature_set:
            d = (p1[i] - p2[i])
            distance += numpy.dot(d, d)

        distance = numpy.sqrt(distance); 
        return distance

    def k_nearst_neighbor(self, test, p, feature_set, k):
        res = []
        for t in test:
            neighbors = Queue.PriorityQueue()
            for pp in p:
                neighbors.put((self.Euclidean_distance(t, pp, feature_set), pp))

            # vote for test data using k nearest neighbors
            vote = [0]*10
            for i in range(k):
                n = neighbors.get()[1];
                vote[int(n[0])] += 1

            res.append(numpy.argmax(vote))

        return res

    def k_fold_cross_validation(self, data, feature_set, k_fold):
        correct = 0
        k = len(data) - k_fold + 1
        for i in range(0, len(data), k):
            test = data[i:i+k]
            train = data[:i]+data[i+k:]
            predict_class = self.k_nearst_neighbor(test, train, feature_set, 1)
            
            for j in range(len(test)):
                if test[j][0] == predict_class[j]:
                    correct += 1

        return float(correct)/len(data)


class feature_selection:
    def __init__(self):
        print "Welcome to Yuan Yao Feature Selection Algorithm."
        file = raw_input("Type in the name of the file to test: ")

        # while True:
        print "\nType the number of the algorithm you want to run."

        print "\t1) Forward Selection"
        print "\t2) Backward Elimination"
        print "\t3) Yuan's Special Algorithm."

        method = raw_input()
        method = int(method)
        # file = "cs_205_NN_datasets/cs_205_small65.txt"
        data = extract_data(file)

        start_time = time.time()
        if method == 1:
            accuracy = self.forward_selection(data)
        elif method == 2:
            accuracy = self.backward_selection(data)
        elif method == 3:
            accuracy = self.special_selection(data)
        else:
            print "Please choose a method"
            
        end_time = time.time()
        time_elapsed = end_time - start_time
        print "Time cost: %fs" %time_elapsed

    def forward_selection(self, data):
        knn = KNN_classifier();
        default = self.default_rate(data)
        print "Using no feature, the default rate is %.1f%%" %(default*100)
        print "\nBeginning search."
        # foward search
        remain_features = [i for i in range(1, len(data[0]))]
        best_features = []
        feature_set = []
        fs = ()
        local_maxima_count = 0
        irrelevant_count = 0
        while remain_features:
            if local_maxima_count > 1:
                print "Break searching since accuracy has decreased many times"
                break
            if irrelevant_count > 1:
                print "Stop searching since accuracy only changes a little due to irrelevant feature"
                best_features.pop()
                break

            for feature in remain_features:
                temp = []
                if fs:
                    temp += fs[1]
                
                temp.append(feature)
                accuracy = knn.k_fold_cross_validation(data, temp, len(data))
                feature_set.append((accuracy, temp))
                print "\tUsing feature(s)", temp, "accuracy is %.1f%%" %(accuracy*100)
                if remain_features.index(feature) == len(remain_features) - 1:
                    print

            # store the best result in best_features
            fs = self.maxSet(feature_set)

            if best_features:
                prev_accurate = best_features[-1][0]
                if fs[0] < prev_accurate:
                    local_maxima_count += 1
                    print "(Warning, Accuracy has decreased! Continuing search in case of local maxima)"
                
                elif fs[0] - prev_accurate < 0.02:
                    irrelevant_count += 1

            print "Feature set", fs[1], "was best, accuracy is %.1f%%" %(fs[0]*100)
            best_features.append(fs)
            feature_set = []
            remain_features.remove(fs[1][-1])

        # report the final result
        res = self.maxSet(best_features)     
        print "\nFinished search! The best feature subset is", res[1], ", which has an accuracy of %.1f%%" %(res[0]*100)
        

    def backward_selection(self, data):
        knn = KNN_classifier()
        default = self.default_rate(data)
        print "Using no feature, the default rate is %.1f%%" %(default*100)
        print "\nBeginning search."
        # foward search
        curr_feature = [i for i in range(1, len(data[0]))]
        feature_set = []
        best_features = []
        fs = () 
        accuracy = knn.k_fold_cross_validation(data, curr_feature, len(data))
        print "\tUsing feature(s)", curr_feature, "accuracy is %.1f%%" %(accuracy*100)

        while len(curr_feature) > 1:
            for feature_to_remove in curr_feature:
                temp = []
                temp += curr_feature
                temp.remove(feature_to_remove)
                accuracy = knn.k_fold_cross_validation(data, temp, len(data))
                feature_set.append((accuracy, temp))
                print "\tUsing feature(s)", temp, "accuracy is %.1f%%" %(accuracy*100)
                if curr_feature.index(feature_to_remove) == len(curr_feature) - 1:
                    print    

            fs = self.maxSet(feature_set)
            if best_features:
                prev_accurate = best_features[-1][0]
                if fs[0] < prev_accurate:
                    print "(Warning, Accuracy has decreased! Continuing search in case of local maxima)"
                
            print "Feature set", fs[1], "was best, accuracy is %.1f%%" %(fs[0]*100)
            best_features.append(fs)
            feature_set = []
            curr_feature = fs[1]    

        # report the final result
        res = self.maxSet(best_features)
        print "\nFinished search! The best feature subset is", res[1], ", which has an accuracy of %.1f%%" %(res[0]*100)
        

    # This search method uses forward search after searching for every pair of the two features
    # to give a much better result
    def special_selection(self, data):
        knn = KNN_classifier();
        default = self.default_rate(data)
        print "Using no feature, the default rate is %.1f%%" %(default*100)
        print "\nBeginning search."
        
        remain_features = [i for i in range(1, len(data[0]))]
        best_features = []
        feature_set = []
        fs = ()
        local_maxima_count = 0
        irrelevant_count = 0

        # search for each pair of the two features
        for i in range(len(remain_features)):
            for j in range(i+1, len(remain_features)):
                pair = [remain_features[i], remain_features[j]]
                accuracy = knn.k_fold_cross_validation(data, pair, 10)
                feature_set.append((accuracy, pair))
                print "\tUsing feature(s)", pair, "accuracy is %.1f%%" %(accuracy*100)
                
        fs = self.maxSet(feature_set)
        print "Feature set", fs[1], "was best, accuracy is %.1f%%" %(fs[0]*100)
        best_features.append(fs)
        feature_set = []
        for p in fs[1]:
            remain_features.remove(p)

        while remain_features:
            if local_maxima_count > 1:
                print "Break searching since accuracy has decreased many times"
                break
            if irrelevant_count > 1:
                print "Stop searching since accuracy only changes a little due to irrelevant feature"
                best_features.pop()
                break

            for feature in remain_features:
                temp = []
                if fs:
                    temp += fs[1]
                
                temp.append(feature)
                accuracy = knn.k_fold_cross_validation(data, temp, len(data))
                feature_set.append((accuracy, temp))
                print "\tUsing feature(s)", temp, "accuracy is %.1f%%" %(accuracy*100)
                if remain_features.index(feature) == len(remain_features) - 1:
                    print

            # store the best result in best_features
            fs = self.maxSet(feature_set)

            if best_features:
                prev_accurate = best_features[-1][0]
                if fs[0] < prev_accurate:
                    local_maxima_count += 1
                    print "(Warning, Accuracy has decreased! Continuing search in case of local maxima)"
                
                elif fs[0] - prev_accurate < 0.02:
                    irrelevant_count += 1

            print "Feature set", fs[1], "was best, accuracy is %.1f%%" %(fs[0]*100)
            best_features.append(fs)
            feature_set = []
            remain_features.remove(fs[1][-1])

        # report the final result
        res = self.maxSet(best_features)     
        print "\nFinished search! The best feature subset is", res[1], ", which has an accuracy of %.1f%%" %(res[0]*100)
        

    def default_rate(self, data):
        counters = [0]*10
        for i in data:
            counters[int(i[0])] += 1
        return float(max(counters))/len(data)

    def maxSet(self, feature_set):
        fs = (0, [])
        for m in feature_set:
            if fs[0] < m[0]:
                fs = m
        return fs


def z_normalized(data):
    means = numpy.mean(data, axis=0, dtype=numpy.float64)
    stds = numpy.std(data, axis=0, dtype=numpy.float64)

    for i in range(len(data)):
        for j in range(1, len(data[i])):
            data[i][j] = (data[i][j] - means[j])/stds[j]


def extract_data(file):
    f = open(file, 'r')
    data = []
    
    line = f.readline()
    while line:
        data.append([float(x) for x in line.split()])
        line = f.readline()

    features = len(data[0])-1
    instances = len(data)
    print "This dataset has %d features (not including the class attribute), with %d instances." %(features, instances)
    print "Please wait while I normalize the data...",
    z_normalized(data);
    print "Done!"

    return data

if __name__ == "__main__":
    feature_set = feature_selection()















