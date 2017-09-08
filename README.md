# RetinaNet
-----------------

`softmax + gradient clipping` version
end2end testing:
[mAP(0.6813)](https://drive.google.com/open?id=0B_qzepxA9F3vSExWMG8xX2x2aUE)

python ./tools/test_net.py --gpu 0 --weights
output/RetinaNet_end2end/voc_0712_trainval/FPN_iter_140000.ckpt
--imdb voc_0712_test --cfg ./experiments/cfgs/RetinaNet_end2end.yml --network
RetinaNet_train_test



end2end training:

nohup ./experiments/scripts/RetinaNet_end2end.sh 0 RetinaNet pascal_voc0712 --set
RNG_SEED 42 TRAIN.SCALES "[600]" > RetinaNet.log 2>&1 &

tail -f RetinaNet.log


------------------------

TODO:
1. implement `sigmoid + special bias initialization`
2. wash up dirty code
