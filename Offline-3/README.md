# Feed-Forward Neural Network

In this assignment we will implement a FNN from scratch and train it to classify letters.

## Dataset

The dataset is can be downloaded with the following code:

```python
import torchvision.datasets as ds

train_validaton_set = ds.EMNIST(root='./data', split='letters', 
                        train=True, 
                        transform=transforms.ToTensor(), 
                        download=True)

test_set = ds.EMNIST(root='./data', split='letters',
                        train=False, 
                        transform=transforms.ToTensor())
```
The dataset contains 28x28 images of letters from the Latin alphabet. Split the train-validation dataset as 85%-15% to form your train set and validation set.


## Preservation of the Trained Model

We must save the final trained model in a file named `model.pth` in the current directory. 

We must write a separate python script named `test_<roll_number>.py` which will load the saved model and print the accuracy on the test set. The script must be run as follows:

```bash
python3 test_<roll_number>.py
```

