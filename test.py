import torch
import torch.nn.functional as F

def test(device, network, test_loader, test_set_len, test_losses, no_record=False):
    network.eval() # 测试模式
    test_loss = 0
    correct = 0
    with torch.no_grad(): # 禁用梯度计算，节约内存
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.argmax(dim=1, keepdim=True) # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item() # 计算正确的数量

    test_loss /= test_set_len
    if no_record is False:
        test_losses.append(test_loss) # 记录测试损失
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{test_set_len} "
          f"({100. * correct / test_set_len:.2f}%)\n")
    
    return test_losses