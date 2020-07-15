This repository contains code for training an LSTM with our derived regularizers on PennTreebank and WikiText-2. There is also code for training a QRNN on WikiText-103. Our code is based off of the following repository: https://github.com/salesforce/awd-lstm-lm. 

This code uses PyTorch version 0.4.1.
 
To train an LSTM with standard dropout:

```python train_lstm.py --data <location of PTB or Wiki-2 dataset> --batch_size 20 --update_type drop_standard --save <where to log> --dropout_reps 1```

The ```--update_type``` argument controls the type of regularization used. The ```--dropout_reps``` argument controls the value of k in DROP_k. Setting ```--update_type jreg_sample_logit``` will use our explicit regularizer only (with coefficients set via the ```exp_regi, exp_rego, exp_regh, exp_regw``` arguments). Setting ```--update_type drop_standard+taylor_fo``` will perform DROP_k updates with our added update noise (with coefficients set via the ```imp_regi, imp_rego, imp_regh, imp_regw``` arguments). Setting ```--update_type jreg_sample_logit+taylor_fo``` will use the combination of our explicit and implicit regularizers. 

To train a QRNN with standard dropout on WikiText-103:

```python qrnn_wiki103.py --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 15 --optimizer adam --lr 0.5e-3 --data <location of Wiki-103 dataset> --lr_decay 0.1 --when 12 --update_type drop_standard --dropout_reps 1 --save_every 25000 --save <directory to save logs>```



