#--------------------------------------------------------------------
팀명 : Telepix 
팀원 : 박재완, 정영상, Hagai
순위 : 19위 

폴더별 실행순서 

00.Data_Aug
01.Baseline_CRNN
02.Clova_Github
03.Abinet
04.Ensemble
05.Post_Processing


#-----------------------------------------------------------------------
data augmentation
00.Data_Aug/0.data_aug_v1.ipynb 참조 


#-------------------------------------------------------------------------
model 1 : resnet50 + rnn

# 학습 & 추론 
01.Baseline_CRNN/[OCR]_[MODEL_1]_res50_rcnn.ipynb 참조 및 실행 

#-------------------------------------------------------------------------
model 2 : resnet152 + rnn 

# 학습 & 추론 
01.Baseline_CRNN/[OCR]_[MODEL_2]_res152_rcnn.ipynb 참조 및 실행 

#-------------------------------------------------------------------------
model 3 : clova_tps_resnet_bilstm_ctc

# 학습코드 
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

# 추론코드 
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

#-------------------------------------------------------------------------
model 4 : clova_tps_vgg_bilstm_ctc

# 학습코드 
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

# 추론코드 
python demo.py \
--sig 1017-7 \
--batch_size 256 \
--Transformation TPS \
--FeatureExtraction VGG \
--SequenceModeling BiLSTM \
--Prediction Attn \
--image_folder /mnt/e/01.Eric/01.Dataset/07.Dacon/test_sharpened/test_sharpened \
--saved_model ../06.Model_Weights/TPS-VGG-BiLSTM-Attn-Seed1017_77999_best_accuracy.pth

#-------------------------------------------------------------------------
model 5 : abinet 
03.Abinet/main.sh 참조



#-------------------------------------------------------------------------
Ensemble 
04.Ensemble/04_0.ensemble_prepare_clova _v3.ipynb 참조 



#-------------------------------------------------------------------------
Post-Processing
05.Post_Processing_1.py 참조 


