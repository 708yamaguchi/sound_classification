Usage
=====

## Quick demo
This is sound classification demo using ThinkPad's build-in camera and microphone. 3 class classification using spectrogram (applause, flick, voice)
```
rosrun sound_classification create_dataset.py            # create dataset from spectrogram
rosrun sound_classification train.py --gpu 0 --epoch 100 # train
roslaunch sound_classification microphone.launch         # classification on ROS
```

![Experiment](https://github.com/708yamaguchi/sound_classification/blob/media/spectrogram_classification_with_thinkpad.gif)


Upper left : Estimated class
Left       : spectrogram
Right      : Video


## Commands
0. Save configs of sound classification in `config/sound_classification.yaml`. (e.g. microphone name, sampling rate, etc)

1. Record noise sound to calibrate microphone (Spectral Subtraction method). The noise sound is recorded in `scripts/mean_noise_sound.npy`.
```
roslaunch sound_classification save_noise_sound.launch
```

2. Save your original spectrogram in `train_data/original_spectrogram`. Specify target object class.
```bash
roslaunch sound_classification save_spectrogram.launch target_class:=(taget object class)
```
NOTE: You can change microphone by giving `microphone_name` argument to this roslaunch. The names of microphones can be seen by `pyaudio.PyAudio().get_device_info_by_index(index)` fuction.

NOTE: You can change threshold of hitting detection by giving `hit_volume_threshold` argument to this roslaunch.

3. Create dataset for training with chainer (Train dataset is augmented, but test dataset is not augmented). At the same time, mean of dataset is calculated. (saved in `train_data/dataset/mean_of_dataset.png`)
```bash
rosrun sound_classification create_dataset.py
```

4. Visualize created dataset (`train` or `test` must be selected as an argument)
```bash
rosrun sound_classification visualize_dataset.py train
```

5. Train with chainer. Results are output in `scripts/result`
```bash
rosrun sound_classification train.py --gpu 0 --epoch 100
```
NOTE: Only `NIN` architecture is available now.

6. Classify spectrogram on ROS. (Results are visualized in rqt)
```bash
roslaunch sound_classification microphone.launch
```

7. Record/Play rosbag
```bash
# record
roslaunch sound_classification record_sound_classification.launch filename:=$HOME/.ros/hoge.bag
# play
roslaunch sound_classification play_sound_classification.launch filename:=$HOME/.ros/hoge.bag
```


Microphone
==========
ThinkPad T460s build-in microphone
