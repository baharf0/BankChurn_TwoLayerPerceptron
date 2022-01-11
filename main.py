
import numpy as np
import pandas as pd

learning_rate = 0.005
nodes_l1 = 16
nodes_l2 = 8
epoch = 6000

data1 = pd.read_csv('Churn_Data.csv', sep=',')
desired = data1['Exited']
no_need_columns = [0, 1, 2, 13]
data2 = data1.drop(data1.columns[no_need_columns], axis=1)
data3 = data2.replace(['Male', 'Female'], [0, 1])
country = data3['Geography'].unique()
country_number = []
for i in range(0, len(country)):
    country_number.append(i)
data4 = data3.replace(country, country_number)
data = data4.apply(lambda x: (x - x.min())/(x.max()-x.min()), axis=0)
train = data.head(int(len(data)*0.8))  # 7200x10
train_desired = desired.head(int(len(data)*0.8))  # 7200x1
test = data.drop(train.index)  # 1800x10
test_desired = desired.drop(train.index)  # 1800x1


def sigmoid(net):
    return 1/(1+np.exp(-net))

# feedforward


bias1 = 1 * np.random.uniform(0, 1)  # 1x1
bias2 = 1 * np.random.uniform(0, 1)  # 1x1
bias3 = 1 * np.random.uniform(0, 1)
output = [[0]] * len(train)  # 7200x1
w1 = np.random.uniform(-1, 1, (len(train.columns), nodes_l1))  # 10x16
h1 = sigmoid(np.dot(train, w1) + bias1)  # 7200x16
w2 = np.random.uniform(-1, 1, (nodes_l1, nodes_l2))  # 16x8
h2 = sigmoid(np.dot(h1, w2) + bias2)  # 7200x8
w3 = np.random.uniform(-1, 1, (nodes_l2, 1))  # 8x1
f_net = sigmoid(np.dot(h2, w3) + bias3)  # 7200x1
for j in range(0, len(f_net)):
    output[j] = 1 if f_net[j] >= 0.5 else 0  # 7200x1
error = train_desired - output  # 7200x1
accurate_count = len(train) - np.count_nonzero(error)  # 7200x1
accuracy = accurate_count / len(train) * 100  # 1x1
print("accuracy before update: ", accuracy)

# backpropagation


w1_new = w1
w2_new = w2
w3_new = w3
h1_new = h1
h2_new = h2
f_net_new = f_net
output_new = [[0]] * len(train)  # 7200x1
for e in range(0, epoch):
    delta1 = [[]]  # 7200x16
    delta2 = [[]]  # 7200x8
    delta3 = [[]]  # 7200x1
    delta3 = f_net_new * (1-f_net_new) * (np.array(train_desired).reshape(len(train), 1) - f_net_new)
    delta2 = h2_new * (1-h2_new) * np.dot(delta3, np.transpose(w3_new))
    delta1 = h1_new * (1 - h1) * np.dot(delta2, np.transpose(w2_new))
    w3_new = w3_new + learning_rate * np.dot(np.transpose(h2), delta3)  # 8x1
    w2_new = w2_new + learning_rate * np.dot(np.transpose(h1), delta2)  # 16x8
    w1_new = w1_new + learning_rate * np.dot(np.transpose(train), delta1)  # 10x16
    h1_new = sigmoid(np.dot(train, w1_new))  # 7200x8
    h2_new = sigmoid(np.dot(h1_new, w2_new))
    f_net_new = sigmoid(np.dot(h2_new, w3_new))  # 7200x1
    for k in range(0, len(f_net)):
        output_new[k] = 1 if f_net_new[k] >= 0.5 else 0
    error_new = train_desired - output_new
    accurate_count_new = len(train) - np.count_nonzero(error_new)
    accuracy_new = accurate_count_new / len(train) * 100
    print("accuracy after", e+1, "update: ", accuracy_new)











