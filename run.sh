#!/bin/bash

ALPHA=0.50
GAMMA=0.50
EPSILON=0.05
python smartcab/agent.py --alpha ${ALPHA} --gamma ${GAMMA} --epsilon ${EPSILON}
