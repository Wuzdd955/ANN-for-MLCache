import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor

def noramalization2(data):
    e_x = np.exp(data - np.max(data,axis=1,keepdims= True))
    return (e_x) / e_x.sum(axis = 1,keepdims= True)

data = pd.read_csv("8trace.csv")

test1 =  pd.read_csv("8s-0.csv")
test2 =  pd.read_csv("8s-1.csv")
test3 =  pd.read_csv("8s-2.csv")
test4 =  pd.read_csv("8s-3.csv")

test1_array = test1.values
test2_array = test2.values
test3_array = test3.values
test4_array = test4.values


array = data.values
scalar = MinMaxScaler()
array = scalar.fit_transform(array)

test1_array = scalar.fit_transform(test1_array)
test2_array = scalar.fit_transform(test2_array)
test3_array = scalar.fit_transform(test3_array)
test4_array = scalar.fit_transform(test4_array)

test1_array = test1_array[:, :80]
test2_array = test2_array[:, :80]
test3_array = test3_array[:, :80]
test4_array = test4_array[:, :80]
X = array[:, :80]

Y = array[:, 80:88]

validation_size = 0.2
seed = 10
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size=validation_size, random_state=seed)
max_socres = 0
min_scores = 0.8577435326459336
bestY  = []
bestY0 = []
bestY1 = []
bestY2 = []
bestY3 = []
for i in range(1):
    for j in range(1):
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size,random_state=j)
        mlp = MLPRegressor(max_iter=200,hidden_layer_sizes=(256,256),learning_rate_init=0.0001,batch_size='auto',early_stopping=False,n_iter_no_change=200)
        #mlp = SVR(kernel='linear', C=0.8, epsilon=0.2,max_iter=300)
        #mlp = DecisionTreeRegressor()
        #mlp = GradientBoostingRegressor(n_estimators=50,max_depth=20,alpha=0.9)
        # start = time.time()
        mlp.fit(X_train, Y_train)
        #time_used = time.time() - start
        #print(time_used)
        K_pred = mlp.predict(X_validation)
        score =r2_score(Y_validation, K_pred)
        if (score > max_socres):
            max_socres = score
            ##print(noramalization2(K_pred))
            #genertated_pred1 = mlp.predict(np.array(genertate_data))
            #genertated_pred2 = mlp.predict(np.array(genertate_data2))
            loss_history = mlp.loss_curve_
            #print(genertated_pred)
            bestY1 = mlp.predict(test1_array)
            bestY2 = mlp.predict(test2_array)
            bestY3 = mlp.predict(test3_array)
            bestY4 = mlp.predict(test4_array)

# loss
dic = {"loss": loss_history}
df = pd.DataFrame(dic)
df.to_excel("loss.xlsx")
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(loss_history)+1), loss_history)
plt.title('MLP Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
print("best_accuracy: ")
print(max_socres)
#print(len(bestY))

bestY1 = noramalization2(bestY1)
bestY1 = pd.DataFrame(bestY1)
pd.concat([pd.DataFrame(test1.values), bestY1], axis=1).to_excel("8s-0.xlsx",index = False,header=False)

bestY2 = noramalization2(bestY2)
bestY2= pd.DataFrame(bestY2)
pd.concat([pd.DataFrame(test2.values), bestY2], axis=1).to_excel("8s-1.xlsx",index = False,header=False)

bestY3 = noramalization2(bestY3)
bestY3 = pd.DataFrame(bestY3)
pd.concat([pd.DataFrame(test3.values), bestY3], axis=1).to_excel("8s-2.xlsx",index = False,header=False)

bestY4 = noramalization2(bestY4)
bestY4 = pd.DataFrame(bestY4)
pd.concat([pd.DataFrame(test4.values), bestY4], axis=1).to_excel("8s-3.xlsx",index = False,header=False)


