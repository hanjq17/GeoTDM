python -u generate_dataset.py --simulation charged --num-train 10000 --num-valid 2000 --num-test 2000 --seed 43 \
--length 5000 --length_test 5000 --sample-freq 100 --n_workers 32

sleep 10

python -u generate_dataset.py --simulation springs --num-train 10000 --num-valid 2000 --num-test 2000 --seed 43 \
--length 5000 --length_test 5000 --sample-freq 100 --n_workers 32

sleep 10

python -u generate_dataset.py --simulation gravity --num-train 10000 --num-valid 2000 --num-test 2000 --seed 43 \
--length 5000 --length_test 5000 --sample-freq 100 --n_workers 32 --n_balls 10
