#!/bin/bash

for i in 272 268 173 249 254 171 259 280 253 250 257 262 247 159 252 277 229 248 235 230
do
  rsync --progress --update -arzvh danielpclin@140.115.70.212:/home/danielpclin/pdl/checkpoint_data02_$i.hdf5 /mnt/c/Users/danielpclin/PycharmProjects/Deep-Learning-Final/
done
