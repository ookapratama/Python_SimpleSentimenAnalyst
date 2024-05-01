import string
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
stop = stopwords.words('indonesian')
import csv
from collections import Counter

# TODO
def load_sent_words_from_file(filename):
  """
  Memuat isi dari sebuah file sentiment words ke sebuah
  dictionary dengan key adalah kata dan value adalah sentiment
  score dari kata tersebut.
  
  separator kata dan score pada file adalah TAB --> "\t"

  walaupun score pada file bertipe integer harap simpan score
  dalam bentuk float.
  
  referensi:
  https://github.com/fajri91/InSet
  
  """

  dict_word_score = {}
  with open(filename, 'r', newline='', encoding='utf-8') as file:
    tsv_reader = csv.reader(file, delimiter='\t')
    for row in tsv_reader:
      dict_word_score.update({row[0]: int(row[1])})
  # print('dictionary file  : ', dict_word_score)

  return dict_word_score

# TODO
def load_sent_words(positive_words, negative_words):
  """
  Menggabungkan dictionary lexicon dari dictionary positive words dan
  negative words. Jika sebuah kata w muncul di kedua dictionary, score
  diganti dengan cara:
  
  score(w) = pos_score(w) + neg_score(w)
  
  Output dari fungsi ini juga merupakan sebuah dictionary dengan
  key adalah kata dan value adalah sentiment score dari kata tersebut.
  """
  # print('positif: ', positive_words)

  # print('negatif: ',negative_words)

  word_score = {}
  # for word, pos_score in positive_words.items():
  #   word_score.append((word, pos_score))
    
  # for word, neg_score in negative_words.items():
  #   found = False
  # for i, (w, s) in enumerate(word_score):
  #   if word == w:
  #       word_score[i] = (w, s + neg_score)
  #       found = True
  #       break
  # if not found:
  #   word_score.append((word, neg_score))
  
  for word, pos_score in positive_words.items():
    word_score[word] = pos_score
    
  for word, neg_score in negative_words.items():
    if word in word_score:
      word_score[word] += neg_score
    else:
      word_score[word] = neg_score
  return word_score

# TODO
def load_stopwords(filename):
  """ memuat semua stopwords dalam bentuk set """
  stopwords = set()
  with open(filename, 'r') as file:
    for line in file:
      stopwords.add(line.strip())
  # print('stopwords: ', stopwords)
  return stopwords
  
#TODO
def load_mobil_dataset(filename):
  """ 
  kita muat data_sentiment_mobil.csv untuk evaluasi.
  setiap baris pada dataset adalah berbentuk:
    
    <kalimat review>;<label>
      
  dimana label berupa {pos, neg} dan delimeter kalimat
  dan label adalah semi-colon ;

  data yang disimpan adalah kalimat beserta labelnya.
  jika menggunakan kesulitan menggunakan list bisa juga
  menggunakan dictionary.
  
  referensi:
  https://ieeexplore.ieee.org/document/8629181
  
  """
  mobil_dataset = []

  with open(filename, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file, delimiter=';')
        for row in csv_reader:
            review, label = row[0], row[1]
            mobil_dataset.append((review, label))

  return mobil_dataset
  
## Preprocessing steps

def prep_lower_case(sentence):
  """ lower casing """
  return sentence.lower()
  
def prep_tokenizer(sentence):
  """ tokenizing """
  return sentence.split()

# TODO
def prep_remove_stopwords(tokenized_sentence, stopwords):
  """ stopword removal """

  results = []
  for sentence in tokenized_sentence:
        filter_sentences = filter(lambda word: word != '', word_tokenize(sentence))
        filter_sentences = [word for word in filter_sentences if word not in stopwords and word]
        results.append(' '.join(filter_sentences))
        
  return results
  
## Prediction
# TODO
def predict(sentence, stopwords, word_score):
  """ fungsi untuk melakukan prediksi orientasi sentiment sebuah kalimat """
  # preprocessing
  lowered_sentence = prep_lower_case(sentence)
  # print('lower :',lowered_sentence)
  tokenized_sentence = prep_tokenizer(lowered_sentence)
  # print('tokenize :',tokenized_sentence)

  tokenized_sentence = prep_remove_stopwords(tokenized_sentence, stopwords)
  # print('stopword : ',tokenized_sentence)
  # print('word scord: ', word_score)

  sentiment_score = 0
  for word in tokenized_sentence:
    if word in word_score:
      # print('score: ',word_score)
      sentiment_score += word_score[word]
  
    # Menentukan sentimen berdasarkan nilai sentimen
  if sentiment_score > 0:
      prediction = "pos"
  elif sentiment_score < 0:
      prediction = "neg"
  else:
      prediction = "netral"
    
  
  # metode untuk prediksi apakah positive atau negative
  # ini hanyalah salah satu metode saja, ada banyak cara lain dalam
  # memanfaatkan sentiment score yang ada pada word_score
  return prediction
## Evaluasi model prediksi

def eval(mobil_dataset, stopwords, word_score):
  """ 
  Fungsi untuk evaluasi seberapa bagus model prediksi yang
  sudah Anda kembangkan di atas.
  
  Metrik evaluasi adalah akurasi, yaitu:
  
      banyaknya kalimat di dataset yang benar diprediksi
      ---------------------------------------------------
                banyaknya kalimat di dataset
  
  """
  true_prediction = 0
  total_sentences = 0
  for sentence, true_label in mobil_dataset:
    predicted_label = predict(sentence, stopwords, word_score)
    if predicted_label == true_label:
      true_prediction += 1
    total_sentences += 1
  return true_prediction / total_sentences
  
#TODO
def word_count_pos(data, stopwords, positive_words):
  # count = 0
  # for sentence, label in data:
  #   tokenized_sentence = prep_tokenizer(prep_lower_case(sentence))
  #   tokenized_sentence = prep_remove_stopwords(tokenized_sentence, stopwords)
  #   for word in tokenized_sentence:
  #     if word in positive_words:
  #       count += 1
  positive_words_count = Counter()
  for sentence, label in data:
      tokenized_sentence = prep_tokenizer(prep_lower_case(sentence))
      tokenized_sentence = prep_remove_stopwords(tokenized_sentence, stopwords)
      for word in tokenized_sentence:
          if word in positive_words:
              positive_words_count[word] += 1
    
    # Mengambil 20 kata positif yang paling sering muncul
  count = positive_words_count.most_common(20)
  print(count)

#TODO
def word_count_neg(data, stopwords, neg_words):
  # count = 0
  # for sentence, label in data:
  #   tokenized_sentence = prep_tokenizer(prep_lower_case(sentence))
  #   tokenized_sentence = prep_remove_stopwords(tokenized_sentence, stopwords)
  #   for word in tokenized_sentence:
  #     if word in neg_words:
  #       count += 1
  negative_words_count = Counter()
  for sentence, label in data:
      tokenized_sentence = prep_tokenizer(prep_lower_case(sentence))
      tokenized_sentence = prep_remove_stopwords(tokenized_sentence, stopwords)
      for word in tokenized_sentence:
          if word in neg_words:
              negative_words_count[word] += 1
    
    # Mengambil 20 kata negatif yang paling sering muncul
  count = negative_words_count.most_common(20)
  print(count)

## Program Utama dengan 4 menu

if __name__ == "__main__":
  positive_words = load_sent_words_from_file("positive.tsv")
  negative_words = load_sent_words_from_file("negative.tsv")
  word_score = load_sent_words(positive_words, negative_words)
  stopwords = load_stopwords("stopwords.txt")
  mobil_dataset = load_mobil_dataset("data_sentiment_mobil.csv")
  
  print(".:Pilih Menu:.")
  print("[1] prediksi orientasi sentiment kalimat")
  print("[2] evaluasi model prediksi")
  print("[3] word count positif")
  print("[4] word count negatif")
  
  valid = False
  while not valid:
    try:
      menu = input("Menu yang dipilih = ")
      if menu == "1":
        print("Tuliskan kalimat review tentang mobil Anda!")
        sentence = input(">> ")
        print("prediksi =", predict(sentence, stopwords, word_score))
      elif menu == "2":
        print("akurasi model prediksi =", eval(mobil_dataset, stopwords, word_score) * 100., "persen")
      elif menu == "3":
        word_count_pos(mobil_dataset, stopwords, positive_words)
      elif menu == "4":
        word_count_neg(mobil_dataset, stopwords, negative_words)
      else:
        raise ValueError("Pilihan menu belum benar. Silakan coba lagi.")
      valid = True
    except ValueError as e:
      print(e)