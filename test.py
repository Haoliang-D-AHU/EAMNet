import os
import json
import sys
import logging
import torch
from PIL import Image
from torchvision import transforms
from net import BottleNeck,EAMNet
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Training with pytorch')
parser.add_argument("--test_dir",default=r'/root/autodl-n2as/prosessed_image/test',type=str,help='type of dataset')
parser.add_argument("--weights_path",default=r'/root/autodl-tmp/weights/model_new-9-of-1000-0.9076206684112549-0.567.pth',type=str,help='weights_path')
parser.add_argument("--json_path",default=r'/root/usenet/class_indices.json',type=str,help='json_path')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def trf():
    data_transform = transforms.Compose(
        [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.20132038, 0.20132038, 0.20132038], [0.24066962, 0.24066962, 0.24066962])])
    return data_transform



def main():
    correct_num = 0
    logging.info("================>>开始预测<<================")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    list1 = [[0] * 3 for _ in range(3)] #存放混淆矩阵
    with open(args.json_path, "r") as f:
        class_indict = json.load(f)
        
    # 加载图像
    for class_ in os.listdir(args.test_dir):
        # 每一个类别的路径
        class_path = os.path.join(args.test_dir,class_)
        all_num = len(os.listdir(class_path))
        for filename in tqdm(os.listdir(class_path)):
            image_path = os.path.join(class_path,filename)
            if not os.path.exists(image_path):
                logging.info("image_path is not exists!!!")
            else:
                image = Image.open(image_path)
                image = trf()(image)
                image = torch.unsqueeze(image, dim=0)
                # 1.model 第一次放入GPU
                create_model = EAMNet(block=BottleNeck,layers=[3,4,6,3],num_classes=3).to(device)
                # 2. model权重文件
                assert os.path.exists(args.weights_path), "file: '{}' dose not exist.".format(args.weights_path)
                # 3.权重文件放入模型中
                create_model.load_state_dict(torch.load(args.weights_path, map_location=device))
                create_model.eval()
                with torch.no_grad():  # 第二次   数据需要放入GPU
                    output = torch.squeeze(create_model(image.to(device))).cpu()
                    
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).numpy() # # predict_cla 3
                    predict_class = class_indict[str(predict_cla)] # 预测的类别,要跟实际的类别进行一个比较
                    
                    if(class_=='AD'):
                        if(predict_class=='AD'):
                            list1[0][0]+=1
                        elif(predict_class=='MCI'):
                            list1[0][1]+=1
                        else:
                            list1[0][2]+=1

                    if(class_=='MCI'):
                        if(predict_class=='AD'):
                            list1[1][0]+=1
                        elif(predict_class=='MCI'):
                            list1[1][1]+=1
                        else:
                            list1[1][2]+=1

                    if(class_=='CN'):
                        if(predict_class=='AD'):
                            list1[2][0]+=1
                        elif(predict_class=='MCI'):
                            list1[2][1]+=1
                        else:
                            list1[2][2]+=1
                    
                    if predict_class == class_:
                        correct_num += 1
        
        logging.info("类别{}的准确率ACC是{}".format(class_, correct_num / all_num))
        print(list1)
        correct_num = 0 
    logging.info("================>>结束预测！<<================")


if __name__ == '__main__':
    main()
