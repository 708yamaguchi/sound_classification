Usage
1;4205;0c=====

## Quick demo
This is sound classification demo using ThinkPad's build-in camera and microphone. 3 class classification using spectrogram (applause, flick, voice)
```
mkdir -p ~/tmp_ws/src
cd ~/tmp_ws/src
git clone https://github.com/708yamaguchi/sound_classification.git
cd ../
catkin build
source ~/tmp_ws/devel/setup.bash
rosrun sound_classification create_dataset.py            # create dataset from spectrogram
rosrun sound_classification train.py --gpu 0 --epoch 100 # train
roslaunch sound_classification microphone.launch         # classification on ROS
```

![Experiment](https://github.com/708yamaguchi/sound_classification/blob/media/spectrogram_classification_with_thinkpad.gif)


Upper left : Estimated class
Left       : spectrogram
Right      : Video


## Commands
0. Download this package and catkin build.
```
mkdir -p ~/tmp_ws/src
cd ~/tmp_ws/src
git clone https://github.com/708yamaguchi/sound_classification.git
cd ../
catkin build
source ~/tmp_ws/devel/setup.bash
```

1. Set configs of sound classification in `config/sound_classification.yaml` (e.g. microphone name, sampling rate, etc). These parameters must not be changed in the following steps.

  You can get list of microphone names by following command.
```
import pyaudio
p = pyaudio.PyAudio()
for index in range(p.get_device_count()):
    print(p.get_device_info_by_index(index)['name'])
```


2. Record noise sound to calibrate microphone (Spectral Subtraction method). The noise sound is recorded in `scripts/mean_noise_sound.npy`. Be quiet during this command.
```bash
roslaunch sound_classification save_noise_sound.launch
```

3. Save your original spectrogram in `train_data/original_spectrogram`. Specify target object class.
```bash
roslaunch sound_classification save_spectrogram.launch target_class:=(taget object class)
```

  NOTE: You can change threshold of hitting detection by giving `hit_volume_threshold` argument to this roslaunch.

4. Create dataset for training with chainer (Train dataset is augmented, but test dataset is not augmented). At the same time, mean of dataset is calculated. (saved in `train_data/dataset/mean_of_dataset.png`)
```bash
rosrun sound_classification create_dataset.py
```

5. Visualize created dataset (`train` or `test` must be selected as an argument)
```bash
rosrun sound_classification visualize_dataset.py train
```

6. Train with chainer. Results are output in `scripts/result`
```bash
rosrun sound_classification train.py --gpu 0 --epoch 100
```
NOTE: Only `NIN` architecture is available now.

7. Classify spectrogram on ROS. Results are visualized in rqt.
```bash
roslaunch sound_classification microphone.launch
```
NOTE: If you don't have enough GPU machine, classification process will be very slow. (In my environment, GeForce 930M is enough.)

8. Record/Play rosbag
```bash
# record
roslaunch sound_classification record_sound_classification.launch filename:=$HOME/.ros/hoge.bag
# play
roslaunch sound_classification play_sound_classification.launch filename:=$HOME/.ros/hoge.bag
```


Microphone
==========
Worked on:
 - ThinkPad T460s build-in microphone
 - MINI Microphone (http://akizukidenshi.com/catalog/g/gM-12864/)
