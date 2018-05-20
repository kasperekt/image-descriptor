#!/bin/sh

if [ -e kubiak_116307_kasperek_116393.zip ]
then
	rm kubiak_116307_kasperek_116393.zip
fi

zip -r kubiak_116307_kasperek_116393.zip \
	autorzy.txt \
	descriptor.py \
	ExposureDescriptor.py \
	HuMomentsDescriptor.py \
	image.py