import matplotlib.pyplot as plt
import numpy as np

def save_result(img, pred_ladmks, save_name):
    '''
    img: PIL Image
    pred_ladmks: 3d array (1 ,70, 2) 
    '''
    assert pred_ladmks.shape == (1,70,2), "pred_ladmks.shape is not (1,70,2)"
    plt.clf()
    plt.imshow(img)
    plt.scatter(pred_ladmks[0, :, 0], pred_ladmks[0, : , 1], s=10, marker='.', c='g')
    plt.savefig(save_name + '.png')

def save_result_std(img, pred_ladmks, save_name):
    '''
    img: PIL Image
    pred_ladmks: 3d array (1 ,70, 3) 
    '''
    assert pred_ladmks.shape == (1,70,3), "pred_ladmks.shape is not (1,70,2)"
    cm = plt.cm.get_cmap('RdYlBu')
    plt.clf()
    plt.imshow(img)
    plt.colorbar(plt.scatter(pred_ladmks[0, :, 0], pred_ladmks[0, : , 1], s=10, marker='.', c=np.exp(pred_ladmks[0, : , 2]), cmap=cm))
    plt.savefig(save_name + '.png')


if __name__ == "__main__":
    import resNet34
    import torchvision
    from PIL import Image
    import torch
    device  = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    
    img_path = "./test_image/FFHQ00002.png"
    img = Image.open(img_path).convert('RGB')
    img = img.resize((256, 256))
    img_tensor = torchvision.transforms.ToTensor()(img).to(device)
                
    model = resNet34.ResNet34(output_param = 3).to(device).eval()
    model.load_state_dict(torch.load("/root/landmark_detection/pretrained/resNet_GNLL_120epoch.pt"))
    with torch.no_grad():
        pred_ladmks = model(img_tensor.unsqueeze(0)).reshape(1, -1 ,3).cpu().numpy()

    ### img = img_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    save_result_std(img, pred_ladmks, "./save_result_std")

    model = resNet34.ResNet34(output_param = 2).to(device).eval() # x, y
    model.load_state_dict(torch.load("/root/landmark_detection/pretrained/resNet_MSE_120epoch.pt"))
    with torch.no_grad():
        pred_ladmks = model(img_tensor.unsqueeze(0)).reshape(1, -1 ,2).cpu().numpy()

    ### img = img_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    
    save_result(img, pred_ladmks, "./save_result")