import pandas as pd
from tqdm import tqdm
from symspellpy_ko import KoSymSpell, Verbosity

import argparse
import os

parser = argparse.ArgumentParser(description='Post Processing for Dacon - Submission')
parser.add_argument('-i', '--input', help='submission.csv file')
parser.add_argument('-o', '--output', help='output filename (.csv)')

"""
python post_processing_2.py -i 'submission/sub_0.resnet50_rnn_AUGCombo_DataAdd_STR_v1_100.csv' -o 'submission/sub_0.resnet50_rnn_AUGCombo_DataAdd_STR_v1_100_post-process2.csv'
"""

root_data_folder = "dataset/open/"
df = pd.read_csv(os.path.join(root_data_folder, 'train.csv'))
corpus = set(df['label'].values)

def main(args):
  submit = pd.read_csv(args.input)
  sym_spell = KoSymSpell()
  sym_spell.load_korean_dictionary(decompose_korean=True, load_bigrams=True)

  result = []
  get_revised_cnt = 0
  suggested_but_all_not_in_corpus = 0
  for index, row in tqdm(submit.iterrows(), total=submit.shape[0]):
    prediction = row['label']
    suggestions = sym_spell.lookup(row['label'], Verbosity.ALL)
    if prediction not in corpus:
      for suggestion in suggestions:
          suggestion_clean = suggestion.term.replace(" ", "").replace(".", "")
          if suggestion_clean not in corpus: continue
          if len(suggestion_clean) != len(prediction): break # I choose break because the next suggestion even getting weirder
          suggested = suggestion_clean
          if suggested != prediction:
            prediction = suggested
            get_revised_cnt += 1
          break
    if prediction == row['label'] and len(suggestions):
      suggested_but_all_not_in_corpus += 1
    result.append(prediction)
    
  submit['label'] = result
  submit.to_csv(args.output, index=False)
  print(f"Done.. Total revised prediction {get_revised_cnt}/{len(submit)} ({(get_revised_cnt/len(submit)):.2f})")
  print(f"Total suggestion but all are not in corpus: {suggested_but_all_not_in_corpus}")

if __name__ == '__main__':
    # modify path to sqlite db
    args = parser.parse_args()
    main(args)