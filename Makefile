TIMESTAMP=$(shell date +%Y%m%d-%H%M%S)

calibrate:
	python main.py --mode calibrate --cameras 0 1 --board-size 6 9 --square-size 260 --params-path data/calibration/$(TIMESTAMP).json

start:
	python main.py --mode run --cameras 0 1
