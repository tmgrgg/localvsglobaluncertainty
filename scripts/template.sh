!python experiments/train_multiswag/experiment.py --dir='/content/drive/My Drive/LocalVsGlobalUncertainty/Experiments/FashionMNIST/DenseNet/2f' \
--name=train_multiswag --dataset=FashionMNIST --model=DenseNet10 --criterion=CrossEntropyLoss \
--batch_size=128 --optimizer=SGD --lr_init=0.1 --lr_final=0.01 --l2=3e-4 --momentum=0.9 \
--training_epochs=150 --swag_epochs=150 --sample_rate=1.0 --rank=90 --num_models=90 --cuda