#Single GPU training (long training period, not recommended)
# python3 tools/train.py -c configs/rec/rec_r45_abinet.yml

#Multi GPU training, specify the gpu number through the --gpus parameter
# python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_r45_abinet.yml

# python3 tools/export_model.py -c configs/rec/rec_r45_abinet.yml -o Global.pretrained_model=./rec_r45_abinet_train/best_accuracy  Global.save_inference_dir=./inference/rec_r45_abinet

python3 tools/infer/predict_rec.py --image_dir='/home/videoo/workspace/dacon/kyowon_ocr/001.datasets/001.raw_dataset/test' --rec_model_dir='./inference/rec/r45_abinet_295ep_raw/' --rec_algorithm='ABINet' --rec_image_shape='3,32,128' --rec_char_dict_path='./ppocr/utils//dict/korean_dict.txt'
