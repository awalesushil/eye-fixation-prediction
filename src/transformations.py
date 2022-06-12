from torchvision.transforms.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomPerspective


# Define transformations for the images
def get_transformer():
    transform = {
    "raw_image": ToTensor(),
    "fixation": ToTensor(),
    "image": Compose([
                    ToTensor(),
                      RandomCrop(224),
                      RandomPerspective(),
                     Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    }
    return transform