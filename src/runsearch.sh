#!/bin/bash
for i in {10..300..10} do
    for j in {10..300..10} do
        for k in {10..300..10} do
            ./search_optimal $i $j $k &
        done
    done
done
wait