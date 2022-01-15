
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Path of file to read
data_path = 'C:/Users/Damjan Denic/OneDrive/Documents/ETF/NM/nba/inputs/CollegeBasketballPlayers2009-2021.csv'
total_ball_data = pd.read_csv(data_path, low_memory = False)
total_ball_data['pick'] = total_ball_data['pick'].fillna(0)
total_ball_data['pick'].loc[(total_ball_data['pick'] > 1)] = 1
#one-hot encoding
total_ball_data = pd.concat([total_ball_data,pd.get_dummies(total_ball_data['yr'],prefix='yr')],axis=1).drop(['yr'],axis=1)
total_ball_data = total_ball_data.drop(['yr_0', 'yr_42.9','yr_57.1', 'yr_None'],axis = 1)
#print(total_ball_data.columns)

# fill NaNs with means
total_ball_data = total_ball_data.fillna(total_ball_data.mean())

# Player names

player_names = total_ball_data[['player_name', 'team']]

# Only include rows in which a player was drafted, and before 2021 (2021 will be used as test data)

ball_data_train = total_ball_data.loc[total_ball_data['year'] < 2021]
ball_data_train_truth = ball_data_train['pick']
ball_data_train = ball_data_train[['GP', 'Min_per', 'Ortg', 'usg', 'eFG',\
       'TS_per', 'ORB_per', 'DRB_per', 'AST_per', 'TO_per', 'FTM', 'FTA',\
       'FT_per', 'twoPM', 'twoPA', 'twoP_per', 'TPM', 'TPA', 'TP_per',\
       'blk_per', 'stl_per', 'ftr', 'porpag', 'adjoe', 'pfr','ast/tov',\
       'drtg', 'adrtg', 'dporpag', 'stops', 'bpm', 'obpm', 'dbpm', 'gbpm', 'mp',\
       'ogbpm', 'dgbpm', 'oreb', 'dreb', 'treb', 'ast', 'stl', 'blk', 'pts', \
       'yr_Sr', 'yr_So', 'yr_Fr', 'yr_Jr']]

ball_data_test = total_ball_data.loc[total_ball_data['year'] == 2021]
ball_data_test_truth = ball_data_test['pick']
ball_data_test = ball_data_test[['GP', 'Min_per', 'Ortg', 'usg', 'eFG',\
       'TS_per', 'ORB_per', 'DRB_per', 'AST_per', 'TO_per', 'FTM', 'FTA',\
       'FT_per', 'twoPM', 'twoPA', 'twoP_per', 'TPM', 'TPA', 'TP_per',\
       'blk_per', 'stl_per', 'ftr', 'porpag', 'adjoe', 'pfr','ast/tov',\
       'drtg', 'adrtg', 'dporpag', 'stops', 'bpm', 'obpm', 'dbpm', 'gbpm', 'mp',\
       'ogbpm', 'dgbpm', 'oreb', 'dreb', 'treb', 'ast', 'stl', 'blk', 'pts', \
       'yr_Sr', 'yr_So', 'yr_Fr', 'yr_Jr']]

print(ball_data_train.shape)


import numpy as np
import torch as T
device = T.device("cuda")  # apply to Tensor or Module

#-----------------------------------------------------------------------

class NbaDataset(T.utils.data.Dataset):

  def __init__(self, ds, ds_truth):
    self.x_data = T.tensor(ds.values.astype(np.float32),
        dtype=T.float32).to(device)
    self.y_data = T.tensor(ds_truth.values.astype(np.float32),
        dtype=T.float32).to(device)
    self.y_data = self.y_data.reshape(-1,1)

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    if T.is_tensor(idx):
      idx = idx.tolist()
    preds = self.x_data[idx,:]  # idx rows, all 4 cols
    lbl = self.y_data[idx,:]    # idx rows, the 1 col
    sample = { 'predictors' : preds, 'target' : lbl }
    # sample = dict()   # or sample = {}
    # sample["predictors"] = preds
    # sample["target"] = lbl

    return sample




def accuracy(model, ds):
  # ds is a iterable Dataset of Tensors
  n_correct = 0; n_wrong = 0

  # alt: create DataLoader and then enumerate it
  for i in range(len(ds)):
    inpts = ds[i]['predictors']
    target = ds[i]['target']    # float32  [0.0] or [1.0]
    with T.no_grad():
      oupt = model(inpts)

    # avoid 'target == 1.0'
    if target < 0.5 and oupt < 0.5:  # .item() not needed
      n_correct += 1
    elif target >= 0.5 and oupt >= 0.5:
      n_correct += 1
    else:
      n_wrong += 1

  return (n_correct * 1.0) / (n_correct + n_wrong)

#----------------------------------------------------------------------

def acc_coarse(model, ds):
  inpts = ds[:]['predictors']  # all rows
  targets = ds[:]['target']    # all target 0s and 1s
  with T.no_grad():
    oupts = model(inpts)         # all computed ouputs
  pred_y = oupts >= 0.5        # tensor of 0s and 1s
  num_correct = T.sum(targets==pred_y)
  acc = (num_correct.item() * 1.0 / len(ds))  # scalar
  return acc

#----------------------------------------------------------------------

def my_bce(model, batch):
  # mean binary cross entropy error. somewhat slow
  sum = 0.0
  inputs = batch['predictors']
  targets = batch['target']
  with T.no_grad():
    oupts = model(inputs)
  for i in range(len(inputs)):
    oupt = oupts[i]
    # should prevent log(0) which is -infinity
    if targets[i] >= 0.5:  # avoiding == 1.0
      sum += T.log(oupt)
    else:
      sum += T.log(1 - oupt)

  return -sum / len(inputs)

#-------------------------------------------------------------------

class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hid1 = T.nn.Linear(48, 96)  # 4-(8-8)-1
    self.hid2 = T.nn.Linear(96, 48)
    self.hid3 = T.nn.Linear(48, 24)
    self.hid4 = T.nn.Linear(24, 8)
    self.oupt = T.nn.Linear(8, 1)

    T.nn.init.xavier_uniform_(self.hid1.weight) 
    T.nn.init.zeros_(self.hid1.bias)
    T.nn.init.xavier_uniform_(self.hid2.weight) 
    T.nn.init.zeros_(self.hid2.bias)
    T.nn.init.xavier_uniform_(self.hid3.weight) 
    T.nn.init.zeros_(self.hid3.bias)
    T.nn.init.xavier_uniform_(self.hid4.weight) 
    T.nn.init.zeros_(self.hid4.bias)
    T.nn.init.xavier_uniform_(self.oupt.weight) 
    T.nn.init.zeros_(self.oupt.bias)

  def forward(self, x):
    z = T.tanh(self.hid1(x)) 
    z = T.tanh(self.hid2(z))
    z = T.tanh(self.hid3(z))
    z = T.tanh(self.hid4(z))
    z = T.sigmoid(self.oupt(z)) 
    return z






# main

#---------------------------------------------------------------------


train_ds = NbaDataset(ball_data_train,ball_data_train_truth)
test_ds = NbaDataset(ball_data_test,ball_data_test_truth)

bat_size = 10
train_ldr = T.utils.data.DataLoader(train_ds,
batch_size=bat_size, shuffle=False)

# 2. create neural network
print("Creating binary NN classifier ")
net = Net().to(device)

# 3. train network
print("\nPreparing training")
net = net.train()  # set training mode
lrn_rate = 0.01
loss_obj = T.nn.BCELoss()  # binary cross entropy
optimizer = T.optim.SGD(net.parameters(),
lr=lrn_rate)
max_epochs = 100
ep_log_interval = 10
print("Loss function: " + str(loss_obj))
print("Optimizer: SGD")
print("Learn rate: 0.01")
print("Batch size: 10")
print("Max epochs: " + str(max_epochs))

print("\nStarting training")
for epoch in range(0, max_epochs):
    epoch_loss = 0.0            # for one full epoch
    epoch_loss_custom = 0.0
    num_lines_read = 0

    for (batch_idx, batch) in enumerate(train_ldr):
        X = batch['predictors']  # [10,4]  inputs
        Y = batch['target']      # [10,1]  targets
        oupt = net(X)            # [10,1]  computeds 

        loss_val = loss_obj(oupt, Y)   # a tensor
        epoch_loss += loss_val.item()  # accumulate
        # epoch_loss += loss_val  # is OK
        # epoch_loss_custom += my_bce(net, batch)

        optimizer.zero_grad() # reset all gradients
        loss_val.backward()   # compute all gradients
        optimizer.step()      # update all weights

    if epoch % ep_log_interval == 0:
        print("epoch = %4d   loss = %0.4f" % \
        (epoch, epoch_loss))
        # print("custom loss = %0.4f" % epoch_loss_custom)
        # print("")
print("Done ")

# ----------------------------------------------------------

# 4. evaluate model
net = net.eval()
acc_train = accuracy(net, train_ds)
print("\nAccuracy on train data = %0.2f%%" % \
(acc_train * 100))
acc_test = accuracy(net, test_ds)
print("Accuracy on test data = %0.2f%%" % \
(acc_test * 100))

# acc_train_c = acc_coarse(net, train_ds)
# print("Accuracy on train data = %0.2f%%" % \
#  (acc_train_c * 100))
# acc_test_c = acc_coarse(net, test_ds)
# print("Accuracy on test data = %0.2f%%" % \
#  (acc_test_c * 100))

# 5. save model
print("\nSaving trained model state_dict \n")
path = "C:\\Users\\Damjan Denic\\OneDrive\\Documents\\ETF\\NM\\nba\\Models\\banknote_sd_model.pth"
T.save(net.state_dict(), path)

# print("\nSaving entire model \n")
# path = ".\\Models\\banknote_full_model.pth"
# T.save(net, path

# print("\nSaving trained model as ONNX \n")
# path = ".\\Models\\banknote_onnx_model.onnx"
# dummy = T.tensor([[0.5, 0.5, 0.5, 0.5]],
#   dtype=T.float32).to(device)
# T.onnx.export(net, dummy, path,
#   input_names=["input1"],
#  output_names=["output1"])

# model = Net()  # later . . 
# model.load_state_dict(T.load(path))

#----------------------------------------------------------------------