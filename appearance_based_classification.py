import numpy as np
from matplotlib import pyplot as plt 
  
def getBins():
    bins = []
    for binElement in range(0, 256, 5):
        bins.append(binElement)
    return bins

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def processHists(data, indices, bins):
    hists = []
    for index in indices:
        image = data[index]
        reds = image[:1024].reshape((32,32)).astype(int)
        greens = image[1024:2048].reshape((32,32)).astype(int)
        blues = image[2048:].reshape((32,32)).astype(int)
        grayscale = np.true_divide(np.add(np.add(reds, greens), blues), 3)
        
        hist, bin_edges = np.histogram(grayscale, bins)
        hists.append(hist)
        # plt.hist(grayscale, bins = bins)
    return hists

def calculateEuclDistance(hist1, hist2):
    dist = np.linalg.norm(hist1-hist2)
    return dist 

def calculateAccuracyCountPerClass(hists_test, automobile_hists, deer_hists, ship_hists, className):
    total_correctly_predicted = 0
    for test_image in hists_test:
        automobile_min = calculateEuclDistance(test_image, automobile_hists[0])
        for training_image in automobile_hists:
            eucl_distance = calculateEuclDistance(test_image, training_image)
            if (eucl_distance < automobile_min):
                automobile_min = eucl_distance
        deer_min = calculateEuclDistance(test_image, deer_hists[0])
        for training_image in deer_hists:
            eucl_distance = calculateEuclDistance(test_image, training_image)
            if (eucl_distance < deer_min):
                deer_min = eucl_distance
        ship_min = calculateEuclDistance(test_image, ship_hists[0])
        for training_image in ship_hists:
            eucl_distance = calculateEuclDistance(test_image, training_image)
            if (eucl_distance < ship_min):
                ship_min = eucl_distance
        min_distance = min(automobile_min, deer_min, ship_min)
        switcher = {
            "automobile": automobile_min,
            "deer": deer_min,
            "ship": ship_min,
        }
        
        if (min_distance==switcher.get(className)):
            total_correctly_predicted = total_correctly_predicted + 1
    return total_correctly_predicted;

def calculateAccuracyRatio(bins):
    #training data
    unPickled = unpickle('C:/Users/Natia_Mestvirishvili/Desktop/UHH/Computer_Vision_I/Assignment_1/cifar-10-python/cifar-10-batches-py/data_batch_1')
    
    data = unPickled['data'.encode()]
    labels = unPickled['labels'.encode()]
    
    automobile_indices = list(filter(lambda x: labels[x] == 1, range(len(labels))))[0:30]
    deer_indices = list(filter(lambda x: labels[x] == 4, range(len(labels))))[0:30]
    ship_indices = list(filter(lambda x: labels[x] == 8, range(len(labels))))[0:30]
    
    automobile_hists = processHists(data, automobile_indices, bins)
    deer_hists = processHists(data, deer_indices, bins)
    ship_hists = processHists(data, ship_indices, bins)
    
    #test data
    unPickled_test = unpickle('C:/Users/Natia_Mestvirishvili/Desktop/UHH/Computer_Vision_I/Assignment_1/cifar-10-python/cifar-10-batches-py/test_batch')
    
    data_test = unPickled_test['data'.encode()]
    labels_test = unPickled_test['labels'.encode()]
    
    automobile_indices_test = list(filter(lambda x: labels_test[x] == 1, range(len(labels_test))))[0:10]
    deer_indices_test = list(filter(lambda x: labels_test[x] == 4, range(len(labels_test))))[0:10]
    ship_indices_test = list(filter(lambda x: labels_test[x] == 8, range(len(labels_test))))[0:10]
    
    automobile_hists_test = processHists(data_test, automobile_indices_test, bins)
    deer_hists_test = processHists(data_test, deer_indices_test, bins)
    ship_hists_test = processHists(data_test, ship_indices_test, bins)
    
    total_correctly_predicted = calculateAccuracyCountPerClass(automobile_hists_test, automobile_hists, deer_hists, ship_hists, "automobile");
    total_correctly_predicted += calculateAccuracyCountPerClass(deer_hists_test, automobile_hists, deer_hists, ship_hists, "deer");
    total_correctly_predicted += calculateAccuracyCountPerClass(ship_hists_test, automobile_hists, deer_hists, ship_hists, "ship");
    total_tested = len(automobile_hists_test) + len(deer_hists_test) + len(ship_hists_test)
    
    ratio_accuracy = total_correctly_predicted/total_tested
    return ratio_accuracy

# ratio with 51 bins
ratio_result = calculateAccuracyRatio(getBins())
print(ratio_result)
# ratio with 2 bins
print(calculateAccuracyRatio([0,127.5,255]))
# ratio with 10 bins
print(calculateAccuracyRatio([0,25.5,51,76.5,102,127.5,153,178.5,204,229.5,255]))
# ratio with 255 bins
bins = []
for binElement in range(0, 256, 1):
   bins.append(binElement)
print(calculateAccuracyRatio(bins))