#!/bin/bash
echo 'Data generation started'
for i in {1..20}
do
    SEED=$(( $RANDOM % 10))
    SHFT=$(( $RANDOM % 100 - 50))
    CMPR=$(( $RANDOM % 5 + 1))
    echo "Data generation with seed $SEED, shift $SHFT, compression $CMPR"
    python generate_synthetic_data.py $SEED $SHFT $CMPR
done
echo 'Data generation terminated'