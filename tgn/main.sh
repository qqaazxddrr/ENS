#!/bin/bash

dataset=$1

chmod 777 tgn_train.sh 
chmod 777 tgn_train_cl.sh
chmod 777 tgn_test.sh
chmod 777 tgn_test_cl.sh
bash tgn_train.sh $dataset
bash tgn_train_cl.sh $dataset
bash tgn_test.sh $dataset
bash tgn_test_cl.sh $dataset
