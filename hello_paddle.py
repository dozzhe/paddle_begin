import paddle
# 现在面临这样一个任务:
# 乘坐出租车的时候，会有一个10元的起步价，只要上车就需要收取。出租车每行驶1公里，需要再支付每公里2元的行驶费用。
# 当一个乘客坐完出租车之后，车上的计价器需要算出来该乘客需要支付的乘车费用。

def calcFee(distance_travelled):
    return 10+2*distance_travelled


for x in [1.0,3.0,5.0,9.0,10.0,20.0]:
    print(calcFee(x))


# 接下来，把问题稍微变换一下，现在知道乘客每次乘坐出租车的公里数，也知道乘客每次下车的时候支付给出租车司机的总费用.
# 但是并不知道乘车的起步价，以及每公里行驶费用是多少。希望让机器从这些数据当中学习出来计算总费用的规则。
x_data = paddle.to_tensor([[1.0],[3.0],[5.0],[9.0],[10.0],[20.0]])
y_data = paddle.to_tensor([[12.0],[16.0],[20.0],[28.0],[30.0],[50.0]])

linear = paddle.nn.Linear(in_features=1,out_features=1)
w_before_opt = linear.weight.numpy().item()
b_before_opt = linear.bias.numpy().item()
print(w_before_opt,b_before_opt)

mse_loss = paddle.nn.MSELoss()
sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001,parameters=linear.parameters())

total_epoch = 5000
for i in range(total_epoch):
    y_predict = linear(x_data)
    loss = mse_loss(y_predict,y_data)
    loss.backward()
    sgd_optimizer.step()
    sgd_optimizer.clear_gradients()

    if i % 1000 == 0:
        print(i,loss.numpy())

print("finish training,loss ={} ".format(loss.numpy()))