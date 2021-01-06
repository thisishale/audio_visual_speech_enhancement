# Face Landmark-based Speaker-Independent Audio-Visual Speech Enhancement in Multi-Talker Environments
Implementation of the audio-visual speech enhancement system described in the paper [Face Landmark-based Speaker-Independent Audio-Visual Speech Enhancement in Multi-Talker Environments](https://arxiv.org/abs/1811.02480) by University of Modena and Reggio Emilia and Istituto Italiano di Tecnologia.

If you are interested in this work check out the [project page](https://dr-pato.github.io/audio_visual_speech_enhancement/).

## Getting Started
### Install requirements
All code is written for Python 3. Create a virtual environment (optional) and install all the requirements running:
```
pip install -r requirements.txt
```
### Usage
The main program is ```av_speech_enhancement.py```. You can get a list of subcommands typing ```av_speech_enhancement.py -h```.  Try ```av_speech_enhancement.py <subcommand> -h``` for more information about a subcommand.
The audio-visual dataset must have the following directory structure:
```
s1
  /audio
	/file1.waw
	/file2.wav
	...
  /video
	/file1.mpg
	/file2.mpg
	...
s2
  /audio
	/file1.wav
	/file2.wav
	...
  /video
	/file1.mpg
	/file2.mpg
	...
...
```
#### Mixed-speech generation
Generate mixed-speech for training, validation and test sets separately:
```
av_speech_enhancement.py mixed_speech_generator
	--data_dir <data_dir>
	--base_speaker_ids <spk1> <spk2> <...>
	[--noisy_speaker_ids <spk1> <spk2> <...>]
	--audio_dir <audio_dir>
	--dest_dir <dest_dir>
	--num_samples <num_samples>
	--num_mix <num_mix>
	--num_mix_speakers <num_mix_speakers> {1,2}
```

av_speech_enhancement.py mixed_speech_generator --data_dir D:\studies\university\thesis\speech_separation_codes\du16\donesomestuff\dataset_grid --base_speaker_ids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 --noisy_speaker_ids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 --audio_dir audio --dest_dir mix\TRAINING_SET --num_samples 200 --num_mix 3 --num_mix_speakers 1

av_speech_enhancement.py mixed_speech_generator --data_dir D:\studies\university\thesis\speech_separation_codes\du16\donesomestuff\dataset_grid --base_speaker_ids 27 28 29 31 --noisy_speaker_ids 27 28 29 31 --audio_dir audio --dest_dir mix\VALIDATION_SET --num_samples 200 --num_mix 3 --num_mix_speakers 1

av_speech_enhancement.py mixed_speech_generator --data_dir D:\studies\university\thesis\speech_separation_codes\du16\donesomestuff\dataset_grid --base_speaker_ids 30 32 33 34 --noisy_speaker_ids 30 32 33 34 --audio_dir audio --dest_dir mix\VALIDATION_SET --num_samples 200 --num_mix 3 --num_mix_speakers 1

The generated files are organized as follow:
```
TRAINING_SET
	    /s1
	       /file1_with_s2_file2.wav
	       /file2_with_s10_file4.wav
	       ...
	    /s2
	       /file1_with_s12_file5.wav
	       /file2_with_s1_file1.wav
	       ...
	...
VALIDATION_SET
	...
TEST_SET
	...
```
#### Audio pre-processing
Compute power-law compressed spectrograms of mixed-speech audio samples. Repeat this operation for training, validation and test sets. Files are saved in NPY format.
```
av_speech_enhancement.py audio_preprocessing
	--data_dir <data_dir>
	--speaker_ids <spk1> <spk2> <...>
	--audio_dir <audio_dir>
	--dest_dir <dest_dir>
	--sample_rate <sample_rate>
	--max_wav_length <max_wav_length>
```

av_speech_enhancement.py audio_preprocessing --data_dir D:\studies\university\thesis\speech_separation_codes\du16\donesomestuff\dataset_grid --speaker_ids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 --audio_dir mix\TRAINING_SET --dest_dir mix\TRAINING_SET --max_wav_length 48000

av_speech_enhancement.py audio_preprocessing --data_dir D:\studies\university\thesis\speech_separation_codes\du16\donesomestuff\dataset_grid --speaker_ids 27 28 29 31 --audio_dir mix\VALIDATION_SET --dest_dir mix\VALIDATION_SET --max_wav_length 48000

#### Video pre-processing
Extract face landmarks from video using Dlib face detector and face landmark extractor. Files are saved in TXT format (each row has 136 values that represents the flattened x-y values of 68 face landmarks).
```
av_speech_enhancement.py video_preprocessing
	--data_dir <data_dir>
	--speaker_ids <spk1> <spk2> <...>
	--video_dir <video_dir>
	--dest_dir <dest_dir>
	--shape_predictor <shape_predictor_file>
	--ext <video_file_extension>
```
the feature file is 75*136 75 is the number of frames. 136 is twice 68. 68 is the coordinates of landmarks. so landmarks is 75 coordinates (68*2). we reshape it to 75*136 and save it as a text file.

av_speech_enhancement.py video_preprocessing --data_dir D:\studies\university\thesis\speech_separation_codes\du16\donesomestuff\dataset_grid --speaker_ids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 29 --video_dir video --dest_dir video --shape_predictor shape_predictor_68_face_landmarks.dat --ext mpg 

av_speech_enhancement.py show_face_landmarks --video D:\studies\university\thesis\speech_separation_codes\du16\donesomestuff\dataset_grid\s2\video\s2_l_bbim3a.mov --shape_predictor shape_predictor_68_face_landmarks.dat    

```<shape_predictor_file>```  contains the parameters of the face landmark extractor model. You can download a pre-trained model file [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).

If you want to check the result of the face landmark extractor type:
```
av_speech_enhancement.py show_face_landmarks
	--video <video_file>
	--fps <fps>
	--shape_predictor <shape_predictor_file>
```

#### Computing Target Binary Masks
Compute TBMs from clean audio samples. For each speaker Long-Term Average Speech Spectrum (LTASS) is computed and then the threshold is applied to all clean audio samples in ```<audio_dir>```.
```
av_speech_enhancement.py tbm_computation
	--data_dir <data_dir>
	--speaker_ids <spk1> <spk2> <...>
	--audio_dir <audio_dir>
	--dest_dir <dest_dir>
	--sample_rate <sample_rate>
	--max_wav_length <max_wav_length>
```

av_speech_enhancement.py tbm_computation --data_dir D:\studies\university\thesis\speech_separation_codes\du16\donesomestuff\dataset_grid --speaker_ids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 --audio_dir audio --dest_dir tbm --sample_rate 16000 --max_wav_length 48000  

#### TFRecords generation
Before training you have to generate TFRecords of mixed-speech dataset. ```<data_dir>/<mix_dir>``` must have three subdirectories named ```TRAINING_SET```, ```VALIDATION_SET``` and ```TEST_SET``` created with ```<mixed_speech_generator>``` subcommand. Pre-computed spectrogram (NPY format) must be located in the same directory of audio file.
Set ```<tfrecords_mode>```  to "fixed" if samples of the dataset all have the same length (as in GRID corpus), otherwise use "var" (as in TCD-TIMIT corpus).
```
av_speech_enhancement.py tfrecords_generator
	--data_dir <data_dir>
	--num_speakers <number_speakers_mixed> {2,3}
	--mode <tfrecords_mode> {fixed,var}
	--dest_dir <dest_dir>
	--base_audio_dir <base_audio_dir>
	--video_dir <video_dir>
	--tbm_dir <tbm_dir>
	--mix_audio_dir <mix_audio_dir>
	--delta <delta_video_feat> {0,1,2]
	--norm_data_dir <normalization_data_dir>
```

av_speech_enhancement.py tfrecords_generator --data_dir D:\studies\university\thesis\speech_separation_codes\du16\donesomestuff\dataset_grid --num_speakers 2 --mode fixed --dest_dir mix\tfrecords --base_audio_dir audio --video_dir video --tbm_dir tbm --mix_audio_dir mix --delta 0 --norm_data_dir mix\norm


#### Training
Train an audio-visual speech enhancement model described. You can choose between VL2M, VL2M_ref, Audio-Visual Concat and Audio-Visual Concat-ref models.
```
av_speech_enhancement.py training
	--data_dir <data_dir>
	--train_set <training_set_subdir>
	--validation_set <validation_set_subdir>
	--exp <experiment_id>
	--mode <tfrecords_mode> {fixed,var}
	--audio_dim <audio_frame_dimension>
	--video_dim <video_frame_dimension>
	--num_audio_samples <num_audio_samples>
	--model <model_selection> {vl2m,vl2m_ref,av_concat_mask,av_concat_mask_ref}
	--opt <optimizer_choice> {sgd,adam,momentum}
	--learning_rate <learning_rate>
	--updating_step <updating_step>
	--learning_decay <learning_decay>
	--batch_size <batch_size>
	--epochs <num_epochs>
	--hidden_units <num_hidden_lstm_units>
	--layers <num_lstm_layers>
	--dropout <dropout_rate>
	--regularization <regularization_weight>
```

av_speech_enhancement.py training --data_dir D:\studies\university\thesis\speech_separation_codes\du16\donesomestuff\dataset_grid --train_set mix\tfrecords\TRAINING_SET --val_set mix\tfrecords\TRAINING_SET --exp 1 --mode fixed --num_audio_samples 1198 --model vl2m --learning_rate 1e-4 --opt adam --batch_size 32 --epochs 100 --hidden_units 250 --layers 5

#### Testing
Test your trained model. Enhanced speech samples and estimated masks are saved in ```<data_dir>/<output_dir>```. Estimated masks are saved  in subdirectories ```<mask_dir>``` of each speaker directory.
```
av_speech_enhancement.py testing
	--data_dir <data_dir>
	--test_set <training_set_subdir>
	--exp <experiment_id>
	--ckp <model_checkpoint>
	--mode <tfrecords_mode> {fixed,var}
	--audio_dim <audio_frame_dimension>
	--video_dim <video_frame_dimension>
	--num_audio_samples <num_audio_samples>
	--output_dir <output_dir>
	--mask_dir <mask_dir>
```
## Reference
If this project is useful for your research, please cite:
```
@inproceedings{morrone2019face,
  title={Face Landmark-based Speaker-Independent Audio-Visual Speech Enhancement in Multi-Talker Environments},
  author={Morrone, Giovanni and Bergamaschi, Sonia and Pasa, Luca and Fadiga, Luciano and Tikhanoff, Vadim and Badino, Leonardo},
  booktitle={2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6900-6904},
  year={2019},
  organization={IEEE}
}
```
