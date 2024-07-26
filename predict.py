import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import json

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        num_features = 25088
    elif checkpoint['arch'] == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_features = 2048
    
    # Rebuild the classifier or fc layer based on the architecture
    if checkpoint['arch'] == 'vgg16':
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_features, checkpoint['hidden_units']),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(checkpoint['hidden_units'], 102),
            torch.nn.LogSoftmax(dim=1))
    elif checkpoint['arch'] == 'resnet50':
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features, checkpoint['hidden_units']),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(checkpoint['hidden_units'], 102),
            torch.nn.LogSoftmax(dim=1))

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    """
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.229, 0.224])
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict(image_path, model, topk=5, device='cpu'):
    model.to(device)
    model.eval()
    
    image = process_image(image_path).to(device)
    with torch.no_grad():
        log_ps = model(image)
        ps = torch.exp(log_ps)
        top_p, top_index = ps.topk(topk, dim=1)
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}  # Invert the class_to_idx dictionary from the model
    top_p = top_p.cpu().numpy().flatten()
    classes = [idx_to_class[idx] for idx in top_index.cpu().numpy().flatten()]
    
    return top_p, classes

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image using a trained network.')
    parser.add_argument('input', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.input, model, args.top_k, device)
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[cls] for cls in classes]
    
    parts = args.input.split('/')
    category_id = parts[-2]
    if category_id in cat_to_name:
        print(f"Actual Category Name: {cat_to_name[category_id]}")
    else:
        print(f"Actual Category ID: {category_id}")
    
    print('Predicted Classes:', classes)
    print('Probabilities:', probs)

if __name__ == '__main__':
    main()
