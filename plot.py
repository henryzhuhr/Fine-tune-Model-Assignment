
import os
import numpy as np
from matplotlib import pyplot as plt

DIR='logs'
for file in os.listdir(DIR):
    if os.path.splitext(file)[-1] not in ['.log']:
        continue
    with open(os.path.join(DIR,file),'r')as f:
        data_list=[]
        filecontent=f.readlines()[2:]
        for item in filecontent:
            datas=item.rstrip().split('\t')
            datas[0]=int(datas[0])
            for i in (1,2,3,4):
                datas[i]=float(datas[i])
            data_list.append(datas)

    x=[d for d in [item[0] for item in data_list]]    
    train_loss=[d for d in [item[1] for item in data_list]]    
    train_acc=[d for d in [item[2] for item in data_list]]    
    valid_loss=[d for d in [item[3] for item in data_list]]    
    valid_acc=[d for d in [item[4] for item in data_list]]
    
    x=np.array(x)
    train_loss=np.array(train_loss)
    train_acc=np.array(train_acc)
    valid_loss=np.array(valid_loss)
    valid_acc=np.array(valid_acc)

    data_dict={        
        'train accuracy':train_acc,
        'valid accuracy':valid_acc,
        'train loss':train_loss,        
        'valid loss':valid_loss,
    }
    plt.figure(figsize=(16, 4.2),dpi=200)
    for i,(name,data) in enumerate(data_dict.items()):
        set_type=name.split(' ')[0]
        data_type=name.split(' ')[1]
        plt.subplot(1,4,i+1)

        plt.plot(x,data)
        plt.xlabel('epoch',fontsize=12)
        plt.ylabel(data_type,fontsize=16)
        plt.xlim(0,100)
        if data_type=='accuracy':
            plt.ylim(0,1)
        plt.title('%s %s'%(set_type,data_type),fontsize=20)
        plt.grid()  # 生成网格
        
    if not os.path.exists('images'):
        os.makedirs('images')
    savefig_name='images/%s.png'%(os.path.splitext(file)[0])

    
    plt.subplots_adjust(wspace=0.35)

    plt.savefig(savefig_name)
    # plt.show()
    plt.close()
