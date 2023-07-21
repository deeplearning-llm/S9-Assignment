
import matplotlib.pyplot as plt
import numpy as np

def imshow_(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    image = np.transpose(npimg, (1, 2, 0))

    return image


def missclassified_plotting(model,device,test_loader):
  model.eval()
  misclassified_images = []
  misclassified_labels = []
  correct_labels = []
  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          misclassified_idx = (pred.squeeze()!= target.squeeze()).nonzero()
          misclassified_images.extend(data[misclassified_idx])
          misclassified_labels.extend(pred[misclassified_idx])
          correct_labels.extend(target[misclassified_idx])

  classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

  cols = 5
  rows = 2
  fig, axs = plt.subplots(rows, cols, figsize=(10,6))
  counter= 0
  rnd_idx = list(range(10))
  for x in range(rows):
    for y in range(cols):
      axs[x, y].set_title(f'Predicted: {classes[misclassified_labels[counter].item()]}\nCorrect: {classes[correct_labels[counter].item()]}')
      axs[x, y].imshow(imshow_(misclassified_images[counter].squeeze()))
      axs[x, y].set_axis_off()
      counter+=1
  plt.tight_layout()
  plt.show() 
