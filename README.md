# RDP-Net
Implemention of RDP-Net using Python's TensorFlow

## *Setup*

1. Clone the directory to your local machine using the following command:
```
    $ git clone https://github.com/dalmouiee/RDP-Net-Ret.git
```

2. Run the create_py_venv.sh script to setup the appropriate python virtual environment: </br>
```
    $ sh scripts\create_py_venv.sh
```
3. Once this script has been executed, the 'py_venv' directory should appear, containing the virtual environment. Type in the following command to activate and switch into it:

    For Windows:
    ```
    $ py_venv\Scripts\activate.bat
    ```
    For Mac/Linux:
    ```
    $ source py_venv/bin/activate
    ```

    You should see (py_venv) added to the beginning of the command line like so:
    ```
    (py_venv) $ 
    ```

    If you wish to exit this virtual environemnt, type:
    ``` 
    (py_venv) $ deactivate
    ```

4. Next, run the get_py_libs.sh script to install the necessary python libraries, needed to run the application locally:

    ```
    (py_venv) $ sh scripts\get_py_libs.sh
    ```
    This may take a few minutes to complete.

### *Training*
 To run training, navigate to the architecture's source code directory and run the training script (refer to the train.py for more information on running the script):

    ```
    (py_venv) $ cd src\prototype\djd
    (py_venv) $ python train.py PATH_TO_TRAINING_DATA
    ```

### *Inference/Prediction*
 To run predict new/test images using a trained model, navigate to the architecture's source code directory and run the predict script (refer to the train.py for more information on running the script):

    ```
    (py_venv) $ cd src\prototype\djd
    (py_venv) $ python predict_model_images.py PATH_TO_TESTING_SET PATH_TO_METAFILE_WITH_ARCHITECTURE_NAME NAME_OF_CHECKPOINT_FILE
    ```

This code was used apart of a study that has yet to be published. The code may be used according to it's MIT license. The data used in the study may be provided upon request. If you are interested, please contact the author via email: d.almouiee@unsw.edu.au