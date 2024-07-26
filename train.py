import argparse
import os
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset and save the model as a checkpoint.")
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='./', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()

    # Print all arguments
    args_dict = vars(args)
    for arg in args_dict:
        print(f"{arg}: {args_dict[arg]}")
    
    # Setup device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print("device:", device)
    
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
        
    # Build and train your network
    model = getattr(models, args.arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Define new classifier/fc layer based on selected architecture
    if args.arch == 'vgg16':
        num_features = 25088
        model.classifier = nn.Sequential(nn.Linear(num_features, args.hidden_units),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(args.hidden_units, 102),
                                         nn.LogSoftmax(dim=1))
    elif args.arch == 'resnet50':
        num_features = 2048
        model.fc = nn.Sequential(nn.Linear(num_features, args.hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(args.hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters() if args.arch == 'resnet50' else model.classifier.parameters(), lr=args.learning_rate)

    # Training loop
    running_loss = 0
    for epoch in range(args.epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model(inputs)
            loss = criterion(logps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            # Validation loop
            model.eval()
            accuracy = 0
            test_loss = 0
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{args.epochs}.. "
                f"Train loss: {running_loss/len(trainloader):.3f}.. "
                f"Validation loss: {test_loss/len(validloader):.3f}.. "
                f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

    # Do validation on the test set
    model.to(device)
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
        else:
            print(f'Accuracy: {(accuracy/len(testloader))*100:.3f}%')
        
    # Save the checkpoint
    model.to ('cpu')
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'arch': args.arch,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hidden_units': args.hidden_units
    }
    # Check if the save directory exists, create it if it does not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pth'))

if __name__ == '__main__':
    main()
