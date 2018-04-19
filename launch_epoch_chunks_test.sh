nohup python fitter_dense_multi.py --overwrite --training-chunks 10 --dev > nohup_10_chunks.out &
nohup python fitter_dense_multi.py --overwrite --training-chunks 50 --dev > nohup_50_chunks.out &
nohup python fitter_dense_multi.py --overwrite --training-chunks 100 --dev > nohup_100_chunks.out &
nohup python fitter_dense_multi.py --overwrite --training-chunks 500 --dev > nohup_500_chunks.out &
nohup python fitter_dense_multi.py --overwrite --training-chunks 1000 --dev > nohup_1000_chunks.out &
