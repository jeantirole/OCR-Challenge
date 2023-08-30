import pandas as pd
from tqdm import tqdm
from hangul_checker import KoreanSpellChecker

import argparse

parser = argparse.ArgumentParser(description='Post Processing for Dacon - Submission')
parser.add_argument('-i', '--input', help='submission.csv file')
parser.add_argument('-o', '--output', help='output filename (.csv)')

"""
example:
python post_processing_1.py -i 'submission/sub_0.resnet50_rnn_AUGCombo_DataAdd_STR_v1_100.csv' -o 'submission/sub_0.resnet50_rnn_AUGCombo_DataAdd_STR_v1_100_post-process1.csv'
"""

def main(args):
  submit = pd.read_csv(args.input)
  ksc = KoreanSpellChecker()
  result = []
  for index, row in tqdm(submit.iterrows(), total=submit.shape[0]):
      result.append(ksc.check_spelling(row['label']).replace(" ", ""))
  submit['label'] = result
  submit.to_csv(args.output, index=False)

if __name__ == '__main__':
    # modify path to sqlite db
    args = parser.parse_args()
    main(args)