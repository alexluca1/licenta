#!/usr/bin/env bash

cat  ../configs/localization/*/*.md > localization_models.md
cat  ../configs/recognition/*/*.md > recognition_models.md
sed -i 's/#/#&/' localization_models.md
sed -i 's/#/#&/' recognition_models.md
sed -i '1i\# Action Localization Models' localization_models.md
sed -i '1i\# Action Recognition Model' recognition_models.md
cat index.rst | grep -q "recognition_models.md"
if [ $? -ne 0 ] ;then
    sed -i '/api.rst/i\   recognition_models.md' index.rst
fi
cat index.rst | grep -q "localization_models.md"
if [ $? -ne 0 ] ;then
    sed -i '/api.rst/i\   localization_models.md' index.rst
fi