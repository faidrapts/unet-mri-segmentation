The code will only run if this folder structure (.py files, dataset folder, figures folder) is preserved. You can see the correct structure in the sync and share shared folder.  
The data_paths.csv file has already been created by running preprocessing.py and is located in the dataset folder on the sync and share.  
  
If you would like to train a model run the train.py file (GPU needed).  
Our best performing model is saved as model_combined.torch and you can directly test (as we tested it) just by
running the test.py file. Everything in the test.py file is set exactly as we used it to test the model's performance.  

The unet.py file contains the U-Net architecture and the dataset class, both defined using PyTorch.
The loss_metrics.py file contains all loss and accuracy metric functions.  
  
In case you encounter any errors or have any further questions contact me at faidra.patsatzi@tum.de :)
