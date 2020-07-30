python experiments/train_model \
--model densenet10 --dataset fashionmnist --dir $1 --name train_model --optimizer SGD --lr_init 0.1 \
--lr_final 0.005 --l2 1e-4 --momentum 0.85 --save_graph --verbose --cuda

