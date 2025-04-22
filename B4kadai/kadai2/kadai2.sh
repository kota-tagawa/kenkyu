#!/bin/bash

for i in $(seq 0 36)
do
    python3 moveviewpoint.py kadai2.jpg input.csv $i
done