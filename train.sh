# python3 train.py --batch_size 256 --model_name Baseline --max_epoch 100
# python3 train.py --batch_size 256 --model_name ModelA --max_epoch 100
# python3 train.py --batch_size 256 --model_name ModelB --max_epoch 100
# python3 train.py --batch_size 256 --model_name ModelC --max_epoch 100
python3 train.py --batch_size 256 --model_name ModelD~256_128 --max_epoch 100
python3 train.py --batch_size 256 --model_name ModelD~256dp_128dp --max_epoch 100
python3 train.py --batch_size 256 --model_name ModelD~1024_512 --max_epoch 200
python3 train.py --batch_size 256 --model_name ModelD~1024dp_512dp --max_epoch 200
python3 train.py --batch_size 128 --model_name ModelD~2048_1024 --max_epoch 200
python3 train.py --batch_size 128 --model_name ModelD~2048dp_1024dp --max_epoch 200