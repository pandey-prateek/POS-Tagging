## How to Load the file

1. If the pos_tagging_lstm.pt is already present pos_tagger.py will load the existing model which is trained on en_atis-ud-train.conllu training set else you need to paste UD_English-Atis folder in the directory in which
    1. en_atis-ud-train.conllu
    2. en_atis-ud-dev.conllu
    3. en_atis-ud-test.conllu

    must be present

2. If the pos_tagging_lstm.pt is not present pos_tagger.py will train a new bi directional LSTM model on on en_atis-ud-train.conllu training set will save an image of the model into pos_tagging_lstm.pt

## How to RUN

1. use the command python3 ./pos_tagger.py to run the file