cd src
mkdir log
mkdir finished

CUDA_VISIBLE_DEVICES=0 python3 -u attack_phase_1.py cw 2.5 1.0 0 &> log/log_cw_2.5a &
CUDA_VISIBLE_DEVICES=1 python3 -u attack_phase_1.py xent 2.5 1.0 0 &> log/log_xent_2.5a &
CUDA_VISIBLE_DEVICES=2 python3 -u attack_phase_1.py cw 2.5 0.2 0 &> log/log_cw_2.5b &
CUDA_VISIBLE_DEVICES=3 python3 -u attack_phase_1.py xent 2.5 0.2 0 &> log/log_xent_2.5b &
CUDA_VISIBLE_DEVICES=4 python3 -u attack_phase_1.py cw 2.5 1.01 1 &> log/log_cw_2.5aq &
CUDA_VISIBLE_DEVICES=5 python3 -u attack_phase_1.py xent 2.5 1.01 1 &> log/log_xent_2.5aq &
CUDA_VISIBLE_DEVICES=6 python3 -u attack_phase_1.py cw 2.5 0.21 1 &> log/log_cw_2.5bq &
CUDA_VISIBLE_DEVICES=7 python3 -u attack_phase_1.py xent 2.5 0.21 1 &> log/log_xent_2.5bq &
wait;

CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 0 &> log/0 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 1 &> log/1 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 2 &> log/2 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 3 &> log/3 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 4 &> log/4 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 5 &> log/5 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 6 &> log/6 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 7 &> log/7 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 8 &> log/8 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 9 &> log/9 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 10 &> log/10 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 11 &> log/11 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 12 &> log/12 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 13 &> log/13 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 14 &> log/14 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 15 &> log/15 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 16 &> log/16 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 17 &> log/17 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 18 &> log/18 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 19 &> log/19 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 20 &> log/20 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 21 &> log/21 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 22 &> log/22 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 23 &> log/23 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 24 &> log/24 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 25 &> log/25 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 26 &> log/26 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 27 &> log/27 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 28 &> log/28 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 29 &> log/29 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 30 &> log/30 &
CUDA_VISIBLE_DEVICES='' python3 -u attack_phase_2.py 31 &> log/31 &
wait;
