#!/bin/bash
root=/cvgl/group/taskonomy/raw/*
src=$PWD
scriptroot=/cvgl2/u/feixia/realenv/data


for model in $root
do
    echo "processing $model" 
    folder=$(basename $model)
    if [ -e $src/$folder ]; then
        echo $src/$folder exists
    elif [ ! -f $model/modeldata/out_res.ply ] || [ ! -f $model/modeldata/sweep_locations.csv ] || [ ! -d $model/dam ]; then
        echo $folder$ does not contain model
    elif [ -e $model/pano ] || [ -e $model/points ]; then
        echo points or pano already exists 
    else
        mkdir $src/$folder
        cp $model/modeldata/sweep_locations.csv $src/$folder/
        cd $model
        /cvgl/software/blender-2.78a/blender -b -noaudio --enable-autoexec --python $scriptroot/generate_points.py --  --NUM_POINTS_NEEDED 2 --MIN_VIEWS 1  --MAX_VIEWS 1 --BASEPATH $model

        node $scriptroot/decode/decode.js --rootdir=/$model --model=$folder
        echo "parse dam file"

        /cvgl/software/blender-2.78a/blender -b -noaudio --enable-autoexec --python $scriptroot/create_rgb_images.py  --  --BASEPATH $model

        /cvgl/software/blender-2.78a/blender -b -noaudio --enable-autoexec --python $scriptroot/create_normal_images.py  --  --BASEPATH $model

        /cvgl/software/blender-2.78a/blender -b -noaudio --enable-autoexec --python $scriptroot/create_mist_images.py  --  --BASEPATH $model

        mv points pano
        mv pano $src/$folder/
        cp -r modeldata $src/$folder

    fi
done
