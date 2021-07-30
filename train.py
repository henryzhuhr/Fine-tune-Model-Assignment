import os
import copy
import tqdm
import torch
from torch import Tensor, nn, optim
import torchvision
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18


# --------------------------------------------------------
#   Args
# --------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description='train')
# attack parameter
parser.add_argument('--model_name', type=str, default='Baseline')
# train parameter
parser.add_argument('--batch_size',type=int, default=32, help='batch size')
parser.add_argument('--num_worker',type=int, default=8, help='number of worker(thread) loading data to device')
parser.add_argument('--max_epoch',type=int, default=100, help='max train epoch')
parser.add_argument('--lr','--learning_rate',type=float, default=0.001, help='learning rate')
args = parser.parse_args()

TRAIN_DIR = 'face_dataset/facescrub_train'
TEST_DIR = 'face_dataset/facescrub_test'
MODEL_NAME = args.model_name
BATCH_SIZES = args.batch_size
LOAD_DATA_THREAD = args.num_worker
LR = args.lr
MAX_EPOCH = args.max_epoch


def Baseline(num_classes: int = 10, pretrained: bool = True):
    model = resnet18(pretrained=pretrained)
    original_state_dict = copy.deepcopy(model.state_dict())
    # ====== fine tune model ======
    # modify the model to make the last feature map size to 8x8x512
    #              Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(
        7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    #                        Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=(
        3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #                                  Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
    model.layer2[0].downsample[0] = nn.Conv2d(
        64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    # =============================
    model.load_state_dict(original_state_dict)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main(model_name: str = 'Baseline',
         max_epoch: int = 100,
         batch_size=8,
         lr=0.001,
         ):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter('runs/%s' % model_name)

    # -------------------------------------
    #   Load dataset
    # -------------------------------------
    train_set = torchvision.datasets.ImageFolder(
        TRAIN_DIR,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
    test_set = torchvision.datasets.ImageFolder(
        TEST_DIR,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
    train_loader = data.DataLoader(train_set, batch_size=batch_size,
                                   shuffle=True, num_workers=LOAD_DATA_THREAD, pin_memory=False)
    test_loader = data.DataLoader(test_set, batch_size=batch_size,
                                  shuffle=False, num_workers=LOAD_DATA_THREAD, pin_memory=False)

    print('[INFO] Load %d class' % (len(train_set.classes)))

    # -------------------------------------
    #   fine tuning model
    # -------------------------------------
    model = Baseline(num_classes=len(train_set.classes),
                     pretrained=True)
    # from torchsummary import summary
    # summary(model,input_size=(3,64,64),device='cpu')
    for name, param in model.named_parameters():
        param.requires_grad = False
    # =============================================

    if model_name == 'Baseline':  # => Baseline
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
    elif model_name == 'ModelA':  # => Model A: fine tune conv5_x
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
        # layer4 / conv5_x
        model.layer4[0].conv1.weight.requires_grad = True
        model.layer4[0].conv2.weight.requires_grad = True
        model.layer4[0].downsample[0].weight.requires_grad = True
        model.layer4[1].conv1.weight.requires_grad = True
        model.layer4[1].conv2.weight.requires_grad = True
    elif model_name == 'ModelB':  # => Model B: fine tune conv4_x conv5_x
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
        # layer3 / conv4_x
        model.layer3[0].conv1.weight.requires_grad = True
        model.layer3[0].conv2.weight.requires_grad = True
        model.layer3[0].downsample[0].weight.requires_grad = True
        model.layer3[1].conv1.weight.requires_grad = True
        model.layer3[1].conv2.weight.requires_grad = True
        # layer4 / conv5_x
        model.layer4[0].conv1.weight.requires_grad = True
        model.layer4[0].conv2.weight.requires_grad = True
        model.layer4[0].downsample[0].weight.requires_grad = True
        model.layer4[1].conv1.weight.requires_grad = True
        model.layer4[1].conv2.weight.requires_grad = True
    elif model_name == 'ModelC':  # => Model C: fine tune conv4_x conv5_x
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
        # conv1_x
        model.conv1.weight.requires_grad = True
        # layer1 / conv2_x
        model.layer1[0].conv1.weight.requires_grad = True
        model.layer1[0].conv2.weight.requires_grad = True
        model.layer1[1].conv1.weight.requires_grad = True
        model.layer1[1].conv2.weight.requires_grad = True
        # layer2 / conv3_x
        model.layer2[0].conv1.weight.requires_grad = True
        model.layer2[0].conv2.weight.requires_grad = True
        model.layer2[0].downsample[0].weight.requires_grad = True
        model.layer2[1].conv1.weight.requires_grad = True
        model.layer2[1].conv2.weight.requires_grad = True
        # layer3 / conv4_x
        model.layer3[0].conv1.weight.requires_grad = True
        model.layer3[0].conv2.weight.requires_grad = True
        model.layer3[0].downsample[0].weight.requires_grad = True
        model.layer3[1].conv1.weight.requires_grad = True
        model.layer3[1].conv2.weight.requires_grad = True
        # layer4 / conv5_x
        model.layer4[0].conv1.weight.requires_grad = True
        model.layer4[0].conv2.weight.requires_grad = True
        model.layer4[0].downsample[0].weight.requires_grad = True
        model.layer4[1].conv1.weight.requires_grad = True
        model.layer4[1].conv2.weight.requires_grad = True
    elif model_name == 'ModelD~256_128':  # [-1, 512, 1, 1]
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),nn.ReLU(),
            nn.Linear(256, 128),nn.ReLU(),
            nn.Linear(128, len(train_set.classes)))
    elif model_name == 'ModelD~256dp_128dp':  # [-1, 512, 1, 1]
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),nn.Dropout(p=0.5),nn.ReLU(),            
            nn.Linear(256, 128),nn.Dropout(p=0.5),nn.ReLU(),            
            nn.Linear(128, len(train_set.classes)))
    elif model_name == 'ModelD~1024_512':  # [-1, 512, 1, 1]
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1024),nn.ReLU(),
            nn.Linear(1024, 512),nn.ReLU(),
            nn.Linear(512, len(train_set.classes)))
    elif model_name == 'ModelD~1024dp_512dp':  # [-1, 512, 1, 1]
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1024),nn.Dropout(p=0.5),nn.ReLU(),            
            nn.Linear(1024, 512),nn.Dropout(p=0.5),nn.ReLU(),            
            nn.Linear(512, len(train_set.classes)))
    elif model_name == 'ModelD~2048_1024':  # [-1, 512, 1, 1]
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 2048),nn.ReLU(),
            nn.Linear(2048, 1024),nn.ReLU(),
            nn.Linear(1024, len(train_set.classes)))
    elif model_name == 'ModelD~2048dp_1024dp':  # [-1, 512, 1, 1]
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 2048),nn.Dropout(p=0.5),nn.ReLU(),            
            nn.Linear(2048, 1024),nn.Dropout(p=0.5),nn.ReLU(),            
            nn.Linear(1024, len(train_set.classes)))
    else:
        raise KeyError('no model name:', model_name)
    # =============================================

    # -------------------------------------
    #   train model
    # -------------------------------------
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    loss_function.to(device)

    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_valid_acc = 0.0
    train_list = []
    model_state_dict_before_train = copy.deepcopy(model).to(device)
    for epoch in range(1, max_epoch + 1):
        print("[Arch] %s [Epoch] %d/%d" % (model_name,epoch, max_epoch))
        running_loss, running__acc = 0, 0
        num_data, step = 0, 0
        model.train()

        pbar = tqdm.tqdm(train_loader)
        for image, label in pbar:
            step += 1
            image: Tensor = image.to(device)
            label: Tensor = label.to(device)
            batch = image.size(0)
            num_data += batch

            output: Tensor = model(image)
            _, pred = torch.max(output, 1)
            loss: Tensor = loss_function(output, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss = loss.item()
            epoch__acc = torch.sum(pred == label).item()
            running_loss += epoch_loss
            running__acc += epoch__acc

            pbar_show = (epoch_loss / batch, epoch__acc / batch)
            pbar.set_description(' Train loss:%.6f  acc:%.6f' % pbar_show)
        train_loss = running_loss / num_data
        train_acc = running__acc / num_data
        print('  Train Loss:%f  Accuracy:%f' % (train_loss, train_acc))
        writer.add_scalar('Train/Loss', train_loss, global_step=epoch)
        writer.add_scalar('Train/Accuracy', train_acc, global_step=epoch)

        # Strat valid
        running_loss, running__acc = 0.0, 0.0
        num_data, step = 0, 0
        model.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(test_loader)
            for image, label in pbar:
                step += 1
                image: Tensor = image.to(device)
                label: Tensor = label.to(device)
                batch = image.size(0)
                num_data += batch

                output: Tensor = model(image)
                _, pred = torch.max(output, 1)
                loss: Tensor = loss_function(output, label)

                epoch_loss = loss.item()
                epoch__acc = torch.sum(pred == label).item()
                running_loss += epoch_loss
                running__acc += epoch__acc

                pbar_show = (epoch_loss / batch, epoch__acc / batch)
                pbar.set_description(' Valid loss:%.6f  acc:%.6f' % pbar_show)
            valid_loss = running_loss / num_data
            valid_acc = running__acc / num_data
            print('  Valid Loss:%f  Accuracy:%f' % (valid_loss, valid_acc))
            writer.add_scalar('Valid/Loss', valid_loss, global_step=epoch)
            writer.add_scalar('Valid/Accuracy', valid_acc, global_step=epoch)
            if valid_acc > best_valid_acc:
                best_model_state_dict = copy.deepcopy(model.state_dict())
                best_valid_acc = valid_acc
        train_list.append(
            [epoch, train_loss, train_acc, valid_loss, valid_acc])
        print()

    if not os.path.exists('logs'):
        os.makedirs('logs')
    with open(os.path.join('logs/diff_%s-b%d-lr%s.txt' % (model_name, batch_size, str(lr).replace('.', '_'))), 'w')as f:
        for n1, p1 in model_state_dict_before_train.named_parameters():
            for n2, p2 in model.named_parameters():
                if n1 == n2 and (p1 - p2).norm(p=2) > 0.001:
                    f.write('Name:%s, Diff: %.4f.\n' %
                            (n1, (p1 - p2).norm(p=2)))

    with open(os.path.join('logs/%s-b%d-lr%s.log' % (model_name, batch_size, str(lr).replace('.', '_'))), 'w')as f:
        f.write('max_epoch=%d,batch_size=%d,lr=%f\n' %
                (max_epoch, batch_size, lr))
        f.write('e\ttrain_loss\ttrain_acc\tvalid_loss\tvalid_acc\n')
        for item in train_list:
            f.write('%d\t%.6f\t%.6f\t%.6f\t%.6f\n' %
                    (item[0], item[1], item[2], item[3], item[4]))

    model.load_state_dict(best_model_state_dict)

    hparam_dict = {'batch size': batch_size, 'lr': lr}
    metric_dict = {'best valid acc': best_valid_acc}
    writer.add_hparams(hparam_dict, metric_dict)
    writer.close()


if __name__ == '__main__':
    print('='*10, MODEL_NAME, '='*10)
    main(model_name=MODEL_NAME,
            max_epoch=MAX_EPOCH,
            batch_size=BATCH_SIZES,
            lr=LR)
