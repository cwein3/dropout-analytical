The code uses PyTorch 0.4.1. To run the standard transformer model on WikiText-103, use the command (this requires 4 GPUs with 11GB memory each)
 
```python train.py --cuda --data <location of wiki103 dataset> --work_dir <where you want to save everything> --dataset wt103 --adaptive --n_layer 16 --d_model 410 --n_head 10 --d_head 41 --d_inner 2100 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 0 --max_step 200000 --tgt_len 150 --mem_len 150 --eval_tgt_len 150 --batch_size 60 --multi_gpu --gpu0_bsz 4 --update_type standard_drop```

To specify the number of samples of dropout noise, use the argument ```--dropout_reps```. To downsample the dataset, use the argument ```--downscale```.

To run our explicit regularizer, use the arguments ```--update_type jac_reg --exp_reg_type jreg_sample_logit --hess_drop 0.1 --hess_dropatt 0```. You might have to specify the command ```--batch_chunk 2``` if memory is an issue.

Our code is based on the following repository: https://github.com/kimiyoung/transformer-xl.
