# nlp_lab3

Exact versions used:  
python==3.11.5    
pandas==2.2.2  
torch==2.2.2+cu118  
torchtext==0.17.2  
tqdm==4.66.1  

# Requirements
```
pip install -r requirements.txt
```
(will not install those exact versions)


# Pipeline

First, if any of the files
```
dataset.csv
vocab.pkl
```
don't exist, create them by using
```
python create_dataset.py
```

Next, if `model.pth` does not exist, create by using
```
python create_model.py
```

Lastly, generate some text using
```
python generate.py > output.txt
```
which outputs result to `output.txt`