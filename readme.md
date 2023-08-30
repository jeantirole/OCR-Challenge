# Summary <br>

OCR Challenge (문자인식)
- 주최사 : 교원그룹
- 플랫폼 : Dacon
- 대회명 : 교원그룹 AI OCR 챌린지
- 대회일정 : 22-12-26 ~ 23-01-16
- Link : https://dacon.io/en/competitions/official/236042/leaderboard
- 참가팀명 : Telepix (3명), 팀리더
- 총참가자 : 1,264명
- 최종순위 : 19위 (상위 5%)
- 개발내용 :
1. Image Augmentation 기법 최적화
2. 다양한 Pre-train Encoder를 활용하기 위한 Custom Modeling
3. 한글 단어에 대한 Post-Processing 파이프라인 개발
4. Decoder에서 추론해내는 단어들에 대한 앙상블 기법 개발
5. Generation model을 활용한 추가 데이터 생성


# Code procedure
00.Data_Aug <br>
01.Baseline_CRNN <br>
02.Clova_Github <br>
03.Abinet <br>
04.Ensemble <br>
05.Post_Processing <br>


# Data augmentation
00.Data_Aug/0.data_aug_v1.ipynb 참조 


# model 1 : resnet50 + rnn
01.Baseline_CRNN/[OCR]_[MODEL_1]_res50_rcnn.ipynb 참조 및 실행 

# model 2 : resnet152 + rnn 
01.Baseline_CRNN/[OCR]_[MODEL_2]_res152_rcnn.ipynb 참조 및 실행 

# model 3 : clova_tps_resnet_bilstm_ctc
cd /02.Clova_Github/
python train.py \
--manualSeed 3333 \
--batch_size 256 \
--train_data ../data/clova_format/clova_format_valid_0.1/train/result \
--valid_data ../data/clova_format/clova_format_valid_0.1/val/result \
--Transformation TPS \
--FeatureExtraction ResNet \
--SequenceModeling BiLSTM \
--Prediction CTC


cd /02.Clova_Github/
python demo.py \
--sig 1017-6 \
--batch_size 256 \
--Transformation TPS \
--FeatureExtraction ResNet \
--SequenceModeling BiLSTM \
--Prediction Attn \
--image_folder ../data/test \
--saved_model ../06.Model_Weights/TPS-ResNet-BiLSTM-Attn-Seed1007_49999_best_accuracy.pth

# model 4 : clova_tps_vgg_bilstm_ctc

cd /02.Clova_Github/
python train.py \
--manualSeed 3334 \
--batch_size 256 \
--train_data ../data/clova_format/clova_format_valid_0.1/train/result \
--valid_data ../data/clova_format/clova_format_valid_0.1/val/result \
--Transformation TPS \
--FeatureExtraction VGG \
--SequenceModeling BiLSTM \
--Prediction CTC

python demo.py \
--sig 1017-7 \
--batch_size 256 \
--Transformation TPS \
--FeatureExtraction VGG \
--SequenceModeling BiLSTM \
--Prediction Attn \
--image_folder /mnt/e/01.Eric/01.Dataset/07.Dacon/test_sharpened/test_sharpened \
--saved_model ../06.Model_Weights/TPS-VGG-BiLSTM-Attn-Seed1017_77999_best_accuracy.pth

# model 5 : abinet 
03.Abinet/main.sh 참조


# Ensemble 
04.Ensemble/04_0.ensemble_prepare_clova _v3.ipynb 참조 


# Post-Processing
05.Post_Processing_1.py 참조 


