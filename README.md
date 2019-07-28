
# Keras implementation of [MPIIGaze dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)

## Requirements
- Python 3.6
- OpenCV
- Keras

## Download and preprocess dataset

    $ ./load-data.sh
    $ python3 gaze-preprocess.py --data PATH_TO_NORMALIZED --out PATH_TO_OUT_FILE
    
## Generate model to json file

    $ python3 gaze-model.py --model PATH_TO_JSON_FILE
    
## Usage

    $ python3 gaze-cnn.py --data PATH_TO_OUT_NPZ_FILE --log PATH_TO_LOG_FILE --model PATH_TO_JSON_FILE_MODEL --epochs EPOCHS_VALUE --batch BATCH_SIZE_VALUE
