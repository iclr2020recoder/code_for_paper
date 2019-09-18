import os
import csv
import re
import sys
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from _collections import defaultdict
import numpy
import random
from _csv import QUOTE_ALL
import html

def get_ngrams(text, n ):
  n_grams = ngrams(word_tokenize(text), n)
  return [ ' '.join(grams) for grams in n_grams]

path = '.'

len_total = 0
len_to_average_total = defaultdict(float)
len_to_reps_total = defaultdict(float)
len_to_reps_ratio_total = defaultdict(float)
total_trigrams = 0
total_trigrams_from_article = 0

print ('Args: side 1 dir, side 2 dir (without /decoded/)')
print ('Entered: %s)' % str(sys.argv))
# The output_ab.txt output is for manual AB testing. The output_ab.csv is for uploading to Amazon Mechanical Turk.
# Sample (test): python abtest.py c:/abisee/log/abtest/decode_test_ref_coverage_lr0.01-ROUGE39.47-len66.1 c:/abisee/log/abtest/decode_test_bvl0.1-ROUGE40.44
# Sample (val):  python abtest.py c:/abisee/log/abtest/decode_val_ref_coverage_lr0.01-ROUGE40.24 c:/abisee/log/abtest/decode_val_bvl0.1-ROUGE41.11-len75.6
 

side_1_path = sys.argv[1]
side_2_path = sys.argv[2]

article_path_test = "c:/abisee/log/output_art/articles_test/"
article_path_val = "c:/abisee/log/output_art/articles_val/"

filename_p = re.compile('([0-9]+)_(reference|decoded)')

with open("output_ab.csv", 'w', encoding="utf-8", newline='') as csvfile:
  csvfile.write('Article,Reference,SideA,SideB,SideAIs,SideBIs\n')
  spamwriter = csv.writer(csvfile, dialect='unix', quoting=QUOTE_ALL)
  with open("output_ab.txt", "w", encoding="utf-8") as output_f:
    
    output_header = False
    coinflips = []
  
    real_side_1_path = side_1_path + "/decoded/"
    real_side_2_path = side_2_path + "/decoded/"
    ref_path = side_1_path + "/reference"
    same = 0
    for file in os.listdir(real_side_1_path):
      side_A_filename = os.path.join(real_side_1_path, file)
      if not os.path.isfile(side_A_filename):
        continue
    
      # Find the corresponding article  
      abs_path = os.path.abspath(side_A_filename)
      if 'decode_test' in abs_path:
        article_path = article_path_test
      elif 'decode_val' in abs_path:
        article_path = article_path_val
      else:
        assert False
        
      article_num = int(filename_p.match(file).group(1))
      if filename_p.match(file) is None:
        print ("Not matched: %s" % file)
      
      output_art_file = os.path.join(article_path, "%06d_article.txt" % article_num)
      with open(output_art_file, "r", encoding="utf-8") as f:
        article_text = f.read()
        
        tokens = article_text.split(' ')
        truncated_article = ' '.join(article_text.split(' ')[:400])
        if len(tokens) > 400:
          truncated_article += ' [...truncated...]'
        article_text = truncated_article.replace(' . ', '.\n').replace('-lrb-', '(').replace('-rrb', ')')
    
      # Read decoded
      with open(side_A_filename, "r", encoding="utf-8") as f:
        side_1_text = f.read()
      
      side_B_name = '%s/%06d_decoded.txt' % (real_side_2_path, article_num) 
      with open(side_B_name, "r", encoding="utf-8") as f:
        side_2_text = f.read()
        
      ref_name = '%s/%06d_reference.txt' % (ref_path, article_num)
      with open(ref_name, "r", encoding="utf-8") as f:
        ref_text = f.read()
    
      #print ("Read Side A: %s" % side_1_text)
      #print ("Read Side B: %s" % side_2_text)
      #print ("Read Article: %s" % article_text)
      
      # Output into file
      if not output_header:
        output_f.write('Articles path: %s\n' % article_path)
        output_f.write('Side 1 path: %s\n' % os.path.abspath(side_1_path))
        output_f.write('Side 2 path: %s\n' % os.path.abspath(side_2_path))
        output_header= True
      coinflip = random.randrange(0, 2)
      if coinflip == 0:
        side_A = side_1_text
        side_B = side_2_text
      else:
        side_A = side_2_text
        side_B = side_1_text
      coinflips.append((article_num, coinflip))
      
      output_f.write('============== ARTICLE %06d ========================================\n\n' % article_num)
      output_f.write('ARTICLE: %s\n\n\n' % article_text)
      output_f.write('--------------------------------------------\n')
      output_f.write('REFERENCE: %s\n\n\n' % ref_text)
      output_f.write('--------------------------------------------\n')
      output_f.write('Side A: %s\n\n' % side_A)
      output_f.write('---------------------------\n')              
      output_f.write('Side B: %s\n\n\n' % side_B)
      
      def escape_for_html(text):
        return html.escape(text).replace('\n', '<br/>')
      
      if side_A != side_B:
        if random.randrange(0, 80) == 0:
          spamwriter.writerow([escape_for_html(article_text), escape_for_html(ref_text), escape_for_html(side_A), escape_for_html(side_B),
                               "1" if coinflip == 0 else "2", "2" if coinflip == 0 else "1"])
      else:
        same += 1

      if article_num % 100 == 0:
        print ("Read article %06d" % article_num)
  
    # Print out coinflips
    print ("%d of %d outputs were identical" % (same, len(coinflips)))
    output_f.write("****************     LEGEND    *******************************\n")
    for (article_num, coinflip) in coinflips:
      side_A_is = "1 (%s)" % side_1_path if coinflip == 0 else "2 (%s)" % side_2_path 
      output_f.write("ARTICLE %06d: Side A is %s\n" % (article_num, side_A_is))
  
    