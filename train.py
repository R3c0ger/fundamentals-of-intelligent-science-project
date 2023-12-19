import torch
import torch.nn.functional as F

def train(epoch, device, network, optimizer, train_loader, log_interval, 
          batch_size_train, train_set_len, train_losses, train_counter, 
          no_record=False, no_show_log=False):
    network.train() # 训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # 将数据移动到 GPU 上
        optimizer.zero_grad()               # 梯度清零
        output = network(data)              # 前向传播
        loss = F.nll_loss(output, target)   # 计算损失
        loss.backward()                     # 反向传播
        optimizer.step()                    # 更新参数
        
        if batch_idx % log_interval == 0:
            if no_show_log is False:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{train_set_len} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

            # 保存训练模型
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')

            if no_record is False:
                train_losses.append(loss.item()) # 记录训练损失
                train_counter.append((batch_idx * batch_size_train) + ((epoch - 1) * train_set_len)) # 记录训练的批次
    
    return train_losses, train_counter