# pipeline and preparation
under deep-text-recognition-benchmark
1. `mkdir data`
2. `python -m venv env`
3. `source env/bin/activate`
3. `pip install -r requirements.txt`
3. `pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git`
3. execute model download segment in demo.ipynb
4. 
`unzip -q data/train.zip -d data/train_orig/ && mv data/train_orig/output/for_recog/train/* data/train_orig/output && rm -r data/train_orig/output/for_recog`
`unzip -q data/test.zip -d data/validation_orig/ && mv data/validation_orig/output/for_recog/test/* data/validation_orig/output && rm -r data/validation_orig/output/for_recog`
`unzip -q data/public_test.zip -d data/test/  && mv data/test/output/for_recog/public_test/* data/test/output && rm -r data/test/output/for_recog`
5. execute prepare.ipynb
6. `python3 create_lmdb_dataset.py --inputPath data/train_orig/ --gtFile data/train_orig/gt.txt --outputPath data/train/`
6. `python3 create_lmdb_dataset.py --inputPath data/validation_orig/ --gtFile data/validation_orig/gt.txt --outputPath data/validation/`

# experiemnt list
6. baseline => sub1.csv
`CUDA_VISIBLE_DEVICES=0 python3 train.py --exp_name TPS-ResNet-BiLSTM-CTC_10000_bsl  \
--train_data data/train/ --valid_data data/validation/ --select_data /  \
--num_iter 2000 --valInterval 100  --batch_size 300  --batch_ratio 1  \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC  \
--FT --saved_model TPS-ResNet-BiLSTM-CTC_10000.pth`
6. keep aspect ratio to prevent heavily distorted images => sub2.csv
`CUDA_VISIBLE_DEVICES=0 python3 train.py --exp_name TPS-ResNet-BiLSTM-CTC_10000_pad  \
--train_data data/train/ --valid_data data/validation/ --select_data /  \
--num_iter 2000 --valInterval 100  --batch_size 300  --batch_ratio 1  \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC  \
--FT --saved_model TPS-ResNet-BiLSTM-CTC_10000.pth --PAD`
7. augment vflip images => sub3.csv
`CUDA_VISIBLE_DEVICES=0 python3 train.py --exp_name TPS-ResNet-BiLSTM-CTC_10000_vflip  \
--train_data data/train/ --valid_data data/validation/ --select_data /  \
--num_iter 2000 --valInterval 100  --batch_size 300  --batch_ratio 1  \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC  \
--FT --saved_model TPS-ResNet-BiLSTM-CTC_10000.pth --PAD`
7. use augment vflip images + adam => sub4.csv
`CUDA_VISIBLE_DEVICES=0 python3 train.py --exp_name TPS-ResNet-BiLSTM-CTC_10000_vflip_adam  \
--train_data data/train/ --valid_data data/validation/ --select_data /  \
--num_iter 2000 --valInterval 100  --batch_size 300  --batch_ratio 1  \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC  \
--FT --saved_model TPS-ResNet-BiLSTM-CTC_10000.pth --PAD --adam --lr 0.001`
7. FAILED, acc too low
`CUDA_VISIBLE_DEVICES=0 python3 train.py --exp_name None-ResNet-None-CTC_10000_vflip_adam  \
--train_data data/train/ --valid_data data/validation/ --select_data /  \
--num_iter 3000 --valInterval 100  --batch_size 300  --batch_ratio 1  \
--Transformation None --FeatureExtraction ResNet --SequenceModeling None --Prediction CTC  \
--FT --saved_model None-ResNet-None-CTC.pth --PAD --adam --lr 0.01`
7. use augment vflip images + adam + enlarge => sub5.csv
`CUDA_VISIBLE_DEVICES=0 python3 train.py --exp_name TPS-ResNet-BiLSTM-CTC_10000_enlarge  \
--train_data data/train/ --valid_data data/validation/ --select_data /  \
--imgW 300 --imgH 50  \
--num_iter 2000 --valInterval 100  --batch_size 100  --batch_ratio 1  \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC  \
--FT --saved_model TPS-ResNet-BiLSTM-CTC.pth --PAD --adam --lr 0.001`

`CUDA_VISIBLE_DEVICES=0 python3 train.py --exp_name TPS-ResNet-BiLSTM-CTC_10000_enlarge  \
--train_data data/train/ --valid_data data/validation/ --select_data /  \
--imgW 300 --imgH 50  \
--num_iter 2000 --valInterval 100  --batch_size 100  --batch_ratio 1  \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC  \
--FT --saved_model saved_models/TPS-ResNet-BiLSTM-CTC_10000_enlarge/iter_1500.pth --PAD --adam --lr 0.001`

`CUDA_VISIBLE_DEVICES=0 python3 train.py --exp_name TPS-ResNet-BiLSTM-CTC_10000_enlarge  \
--train_data data/train/ --valid_data data/validation/ --select_data /  \
--imgW 300 --imgH 50  \
--num_iter 2000 --valInterval 100  --batch_size 100  --batch_ratio 1  \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC  \
--FT --saved_model saved_models/TPS-ResNet-BiLSTM-CTC_10000_enlarge/iter_3700.pth --PAD --adam --lr 0.001`

8. use augment vflip, sharp, blur images + adam => sub6.csv
`CUDA_VISIBLE_DEVICES=0 python3 train.py --exp_name TPS-ResNet-BiLSTM-CTC_sharp_blur  \
--train_data data/train/ --valid_data data/validation/ --select_data /  \
--num_iter 37000 --valInterval 100  --batch_size 300  --batch_ratio 1  \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC  \
--FT --saved_model TPS-ResNet-BiLSTM-CTC.pth --PAD --adam --lr 0.001`

9. add scheduler and correct vflip to vflip+hflip for faster training
CUDA_VISIBLE_DEVICES=0 python3 train.py --exp_name TPS-ResNet-BiLSTM-CTC_test  \
--train_data data/train/ --valid_data data/validation/ --select_data /  \
--num_iter 37000 --valInterval 100  --batch_size 300  --batch_ratio 1  \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC  \
--FT --saved_model TPS-ResNet-BiLSTM-CTC.pth --PAD --adam --lr 0.005









# TEST
`CUDA_VISIBLE_DEVICES=0 python3 train.py --exp_name TPS-ResNet-BiLSTM-CTC_test  \
--train_data data/train/ --valid_data data/validation/ --select_data /  \
--num_iter 37000 --valInterval 100  --batch_size 300  --batch_ratio 1  \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC  \
--FT --saved_model TPS-ResNet-BiLSTM-CTC.pth --PAD --adam --lr 0.01`