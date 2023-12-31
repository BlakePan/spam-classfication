# SMS Spam Classification
Using bert-based model for SMS Spam Classification

# Download Dataset
The SMS Spam Data is already downloaded and saved in the folder `data`.<br>
You can find the source of the dataset below link:<br>
https://archive.ics.uci.edu/dataset/228/sms+spam+collection 

# Install Environment
It is recommended to build a virtual Python env by using Anaconda and run the below command:
```
$ pip install -r requirements.txt
```

# Training
To train `bert-base-uncased` for spam classification by running:
```
$ python train.py -c config-bert.yaml
```

To train `distilbert-base-uncased` for spam classification by running:
```
$ python train.py -c config-distilbert.yaml
```
You can modify hyperparameters in those config files.<br><br>

To enable the Tensorboard writer, please add `-tb` arg in the training command.<br>
For example:
```
$ python train.py -c config-bert.yaml -tb
```


When training starts, you should be able to see the progress and metrics on the screen:
![image](https://github.com/BlakePan/spam-classfication/assets/9764354/fcabdb53-e39d-4bf1-b52b-9c0ba9283015)

The experiment was run on colab, you can check the notebook in the below link:
https://colab.research.google.com/drive/1QmWNf6Fo46Qbw0beCvUUJmyQZNZD2bs0?usp=sharing

# Tensorboard
The training and validation logs are saved in `runs` folder by default,<br>
you can run a Tensorboard service to compare different experiment results by using the below command:
```
$ tensorboard --logdir runs/
```

And click the URL shown on the screen
![image](https://github.com/BlakePan/spam-classfication/assets/9764354/8be507a0-24aa-4e74-b5ba-9f3a9c0d88a6)

Then, you should be able to see the Tensorboard on the browser
<img width="1440" alt="image" src="https://github.com/BlakePan/spam-classfication/assets/9764354/3ea14fda-aa20-4bdc-9226-9c01f5177225">


# Download Fine-tuned model
In this project, we also provided fine-tuned model weights.<br>
You can download those files by using the below commands and try demo directly without training:
```
$ wget –no-check-certificate 'https://drive.usercontent.google.com/download?id=1-xCqfqJqOouxPQmeyvtowa9Cg64lAZdm&export=download&authuser=0&confirm=t&uuid=3b24a51f-e24d-4c46-b7fa-52a660109a4d&at=APZUnTV-bGoABfutQpN4NcWsFmjG:1692866582619' -O models.zip
$ unzip models.zip
```

# Demo
When models are ready, you can run the demo by the below command:<br>
⚠️ If you fine-tuned your own model, please remember to modify the path in `config-demo.yaml`
```
$ python demo.py
```
And click the URL shown on the screen
![image](https://github.com/BlakePan/spam-classfication/assets/9764354/fe9c22c7-87d7-46a5-97d7-251cb95659cc)

Then, you should be able to see the demo webui on the browser
<img width="1440" alt="image" src="https://github.com/BlakePan/spam-classfication/assets/9764354/576f6fbd-da87-44c0-bf78-c81f7cd182c5">

Here are some samples from the validation set that you could try for the demo.
### Hams
0: how come?<br>
1: loosu go to hospital. de dont let it careless.<br>
2: hi my email address has changed now it is<br>
3: k will do, addie & amp ; i are doing some art so i'll be here when you get home<br>
4: aiyo please u got time meh.<be>

<img width="1283" alt="image" src="https://github.com/BlakePan/spam-classfication/assets/9764354/9d62c431-bcca-41d4-84e2-e856d3373460">


### Spams
0: recpt 1 / 3. you have ordered a ringtone. your order is being processed...<br>
1: december only! had your mobile 11mths +? you are entitled to update to the latest colour camera mobile for free! call the mobile update co<br>
2: ree entry in 2 a weekly comp for a chance to win an ipod. txt pod to 80182 to get entry ( std<br>
3: sms services for your inclusive text credits pls gotto www. comuk. net login 3qxj9 unsubscribe with<br>
4: free for 1st week! no1 nokia tone 4 ur mob every week just txt nokia to 8007 get txting and tell ur mates www.<br>

<img width="1274" alt="image" src="https://github.com/BlakePan/spam-classfication/assets/9764354/28da69a7-d2b4-4ce1-95ee-f75f84017c7d">

