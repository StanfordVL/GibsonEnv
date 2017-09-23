#!/bin/bash
root=/cvgl/group/taskonomy/raw/*
src=$PWD

for model in $root
do
    echo "processing $model" 
    folder=$(basename $model)
    if [ -e $src/$folder ]; then
        echo $src/$folder exists
    elif [ ! -f $model/modeldata/out_res.ply ] || [ ! -f $model/modeldata/sweep_locations.csv ]; then
        echo $folder$ does not contain model
    elif [ -e $model/pano ] || [ -e $model/points ]; then
        echo points or pano already exists 
    else
        mkdir $src/$folder
        cp $model/modeldata/sweep_locations.csv $src/$folder/
        #skyboxes=$model/img/low/*skybox0*
        #for skybox in $skyboxes
        #do
        #    name=$(basename ${skybox})
        #    name=../../../..$model/img/low/${name::-5}
        #    echo $name
        #
        #    while [ $(($(ps -e | grep blender | wc -l)+$(ps -e | grep cube2sphere | wc -l))) -gt 20 ]; do 
        #        sleep 0.3
        #        echo waiting 
        #    done 
       
        #    (cube2sphere ${name}1.jpg ${name}3.jpg ${name}4.jpg ${name}2.jpg ${name}0.jpg ${name}5.jpg -r 512 256 -fPNG  -o $folder/$(basename ${skybox} .jpg) )&
    
        #done
        cd $model
        /cvgl/software/blender-2.78a/blender -b -noaudio --enable-autoexec --python /cvgl2/u/feixia/representation-learning-data/scripts/generate_points.py --  --NUM_POINTS_NEEDED 2 --MIN_VIEWS 1  --MAX_VIEWS 1

        /cvgl/software/blender-2.78a/blender -b -noaudio --enable-autoexec --python /cvgl2/u/feixia/representation-learning-data/scripts/create_rgb_images.py  --  


        /cvgl/software/blender-2.78a/blender -b -noaudio --enable-autoexec --python /cvgl2/u/feixia/representation-learning-data/scripts/create_normal_images.py  --  


        /cvgl/software/blender-2.78a/blender -b -noaudio --enable-autoexec --python /cvgl2/u/feixia/representation-learning-data/scripts/create_mist_images.py  -- 

        #wait

        mv points pano
        mv pano $src/$folder/

    fi
done
