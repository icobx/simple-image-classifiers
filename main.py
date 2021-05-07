import os
import random
import cv2
import torch

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import plotly.graph_objects as go 

from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torch_model_helper import dataset, Net
from pathlib import Path
from pandas.core.frame import DataFrame
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn import preprocessing, model_selection, metrics
from sklearn.svm import SVC
from skimage.feature import local_binary_pattern
from PIL import Image
from sklearn.metrics import confusion_matrix

 



preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=160),  # Pre-trained model uses 160x160 input images
    transforms.ToTensor(),
])

device = torch.device('cpu')
# Create an inception resnet (in eval mode):
resnet = None


def load_annotations(path='../4/annotations/identity_CelebA.txt') -> dict:
    annotations = {}

    with open(path, 'r') as fn:
        while True:
            line = fn.readline()

            if not line:
                break
        
            key, value = line.split()
            annotations[key] = int(value)

    return annotations


def load_attrib_annotations(df_emb, read_path='../4/annotations/list_attr_celeba_fwf.txt', write_path='attr_annotations.csv'):
    if Path(write_path).is_file():
        return pd.read_csv(write_path)

    columns_text = 'img 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young'
    columns = columns_text.split()

    attribs = []
    with open(read_path, 'r') as fn:
        while True:
            line = fn.readline()

            if not line:
                break
            
            ls = line.split()
            line = [ls[0], *[int(v) for i, v in enumerate(ls) if i > 0]]

            attribs.append(line)
        
    df_full = pd.DataFrame(attribs, columns=columns)

    df_full.set_index('img', inplace=True)

    df_attr_emb = df_full.loc[df_emb['0'].values]
    df_attr_emb.reset_index(inplace=True)
    df_attr_emb.to_csv(write_path, index=False)

    return df_attr_emb
    

def load_embeddings(emb_path='embeddings.csv', deep=True, lbp_method='nri_uniform', radius=1) -> DataFrame:    
    global resnet

    if Path(emb_path).is_file():

        df = pd.read_csv(emb_path)
        print(df.shape)
        # exit()
        return df

    if deep and not resnet:
        resnet = InceptionResnetV1(pretrained='vggface2').eval()

    d = '../3/data/processed'
    files = os.listdir(d)
    n = len(files)
    embeddings = []
    for i, file in enumerate(files):
        if not Path(f'{d}/{file}').is_file():
            continue

        print(f'calculating embeddings... {(i/n)*100:.2f}%')
        
        if deep:
            embeddings.append([file, *calc_embedding(file, resnet, device)])
        else:
            embeddings.append([file, *calc_lbp(file, radius=radius, method=lbp_method)])

    df_embeddings = pd.DataFrame(data=embeddings)
    df_embeddings.to_csv(emb_path, index=False)

    return df_embeddings


def load_img(filename, to_rgb=False, to_gray=False, dir_path='../3/data/processed'):
    path = f'{dir_path}/{filename}'
    img = cv2.imread(path)

    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def calc_embedding(filename, resnet, device, dir_path='../3/data/processed') -> np.array:
    img = load_img(filename, dir_path=dir_path, to_rgb=True)

    tensor = preprocess(img)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    return resnet(tensor).detach().numpy()[0]

def calc_lbp(filename, dir_path='../3/data/processed', radius=3, method='uniform'):
    img = load_img(filename, dir_path=dir_path, to_gray=True)
    
    # settings for LBP
    n_points = 8 * radius

    lbp_orig = local_binary_pattern(img, n_points, radius, method)
    n_bins = int(lbp_orig.max() + 1)
    hist, _ = np.histogram(lbp_orig, density=True, bins=n_bins, range=(0, n_bins))

    return hist
    

def get_datasets(df_emb: DataFrame, df_attr_anno: DataFrame, attr_col='Male'):
    data_emb = df_emb.values
    data_attr = df_attr_anno[attr_col].apply(lambda x: 1 if x > 0 else 0).values

    X = np.array(data_emb[:, 1:], dtype=np.float32)
    Y = np.array(data_attr, dtype=np.float32)

    print(X.shape, Y.shape)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=1)
    return x_train, x_test, y_train, y_test, X, Y


def get_deep_model(input_shape, learn_rate, lbp=False):
    model = Net(input_shape)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learn_rate)

    return model, optimizer, criterion


def get_dataloaders(df_emb, df_attr_anno, col, val=False):
    # train test split
    x_train, x_test, y_train, y_test, x, y = get_datasets(df_emb, df_attr_anno, col)
    if val:
        x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=0.25, random_state=1)
    # 
    training_data = dataset(x_train, y_train)
    # test_data = dataset(x_test, y_test)
    if val:
        val_data = dataset(x_val, y_val)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    if val:
        val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)
        return train_dataloader, val_dataloader, x_test, y_test, x, y

    return train_dataloader, x_test, y_test, x, y


def training(df_X, df_Y, col='Male', epochs=300, lr=0.01, max_no_improv=15, model_path='models/model.pth', train_data_path='data/training_data.csv'):
    # train_dataloader, x_test, y_test, x, y = get_dataloaders(df_X, df_Y, col)
    train_dataloader, val_dataloader, x_test, y_test, x, y = get_dataloaders(df_X, df_Y, col, val=True)
    input_shape = train_dataloader.dataset.x.shape[1]
    model, opt, crit = get_deep_model(input_shape, lr)

    # train_losses = []
    # valid_losses = []
    # accuracies = []
    training_data = []
    min_valid_loss = np.inf
    n_no_improv = 0
    for i in range(epochs):
        # training
        train_loss = 0.0
        model.train()
        for x_train, y_train in train_dataloader:
            opt.zero_grad()

            output = model(x_train)

            loss = crit(output, y_train.reshape(-1, 1))  # -1 means: calc based on the other param

            predicted = model(torch.tensor(x, dtype=torch.float32))
            acc = (predicted.reshape(-1).detach().numpy().round() == y).mean()

            # opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss = loss.item() * x_train.size(0)
            

        valid_loss = 0.0
        model.eval()
        for x_val, y_val in val_dataloader:
            output = model(x_val)

            loss = crit(output, y_val.reshape(-1, 1))

            valid_loss = loss.item() * x_val.size(0)
        
        tl = train_loss / len(train_dataloader)
        vl = valid_loss / len(val_dataloader)
        print(f'Epoch {i+1}\t| Accuracy: {acc} \
            | Training Loss: {tl} \
            | Validation Loss: {vl}')

        training_data.append([acc, tl, vl])
        
        if min_valid_loss > vl:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}\
                --->{vl:.6f}) \t Saving The Model')
            min_valid_loss = vl
            
            # Saving State Dict
            model.save(model_path)
            n_no_improv = 0
        
        else:
            n_no_improv += 1

            if n_no_improv >= max_no_improv:
                print(f'Early stopping after {n_no_improv} epochs with no improvement on epoch {i}.')
                break

    df_training_data = pd.DataFrame(data=training_data, columns=['accuracy', 'training_loss', 'validation_loss'])
    df_training_data.to_csv(train_data_path)


def test_model(x_test, y_test, model_path='models/model.pth'):

    model = Net(x_test.shape[1])
    model.load(model_path)

    pred = model(torch.tensor(x_test, dtype=torch.float32)).reshape(-1).detach().numpy().round()
    acc = (pred == y_test).mean()

    print(f'Model accuracy: {acc:.4f}')
    return pred


def get_svm_model(x_train, y_train):
    model = SVC(kernel='linear')

    model.fit(x_train, y_train)

    return model


def run_cam(thresh=0.5):
    global resnet

    if not resnet:
        resnet = InceptionResnetV1(pretrained='vggface2').eval()

    mtcnn = MTCNN(keep_all=True, device=device)
    model_male = Net(512)
    model_male.load('models/model-wreg-Male.pth')
    model_attr = Net(512)
    model_attr.load('models/model-wreg-Attractive.pth')

    capture = cv2.VideoCapture(0)
    if not capture.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    while True:
        _, frame = capture.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        img = Image.fromarray(frame)

        # Get cropped and prewhitened image tensor
        imgs_cropped = mtcnn(img)
        boxes, probs = mtcnn.detect(img, landmarks=False)

        # Calculate embedding (unsqueeze to add batch dimension)
        if imgs_cropped is not None and boxes is not None:
            indexes = []
            for i, box in enumerate(boxes):
                x, y, xw, yh = box.astype(int)

                if probs[i] < 0.9:
                    continue

                cv2.rectangle(frame, (x, y), (xw, yh), (0, 255, 0), 2)
                emb = resnet(imgs_cropped[i].unsqueeze(0))
                # print(emb)
                r_male = model_male(emb).detach().cpu().numpy()[0, 0]
                r_attr = model_attr(emb).detach().cpu().numpy()[0, 0]
                print(r_male, r_attr)
            
                if r_attr < thresh:
                    res = 'unattractive'
                else:
                    res = 'attractive'

                if r_male < thresh:
                    res = f'{res} female'
                else:
                    res = f'{res} male'

                
                cv2.putText(frame, f'result: {res}', (xw-50, yh+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Face detection w binary classification', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        
    capture.release()
    cv2.destroyAllWindows()


def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)

    # cm : confusion matrix list(list)
    # labels : name of the data list(str)
    # title : title for the heatmap
    data = go.Heatmap(z=cm, y=labels, x=labels)
    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            text_color = 'white' if value < 500 else 'black'
            annotations.append(
                {
                    "x": labels[i],
                    "y": labels[j],
                    "font": {"color": text_color},
                    "text": str(value),
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False
                }
            )
    layout = {
        "title": title,
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"},
        "annotations": annotations,
        "height": 600,
        "width": 600
    }
    fig = go.Figure(data=data, layout=layout)
#     fig.layout.height = 500
# fig.layout.width = 500
    fig.show()
    # return fig


def main():
    df_emb = load_embeddings(emb_path='embeddings/embeddings.csv')
    # df_emb = load_embeddings(emb_path='embeddings-lbp-nri_uni.csv', deep=False, lbp_method='nri_uniform')
    df_attr_anno = load_attrib_annotations(df_emb)

    col = 'Male'
    # # col = 'Young'
    # # col = 'Attractive'

    model_path = f'models/model-wreg-{col}.pth'
    # model_path = f'models/model-wreg-lbp-{col}-nri.pth'

    # training(df_emb, df_attr_anno, col=col, model_path=model_path, train_data_path=f'data/training_data-wreg-{col}-new.csv')
    _, x_test, _, y_test, _, _ = get_datasets(df_emb, df_attr_anno, col)
    y_pred = test_model(x_test, y_test, model_path=model_path)

    # ###### svm ########################################
    # x_train, x_test, y_train, y_test, x, y = get_datasets(df_emb, df_attr_anno, col)
    # model = get_svm_model(x_train, y_train)
    # y_pred = model.predict(x_test)
    # # print(y_pred)

    # # print(f'for col: {col}')
    # # print(metrics.classification_report(y_test, y_pred))
    
    # attr_labels = ['unattractive', 'attractive']
    # male_labels = ['female', 'male']
    # # show_plot()
    # plot_confusion_matrix(y_test, y_pred, male_labels, f'Confusion matrix: SVM - LBP emb - {col}')

    run_cam()

if __name__ == '__main__':
    main()