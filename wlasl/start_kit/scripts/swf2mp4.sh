#!/bin/bash

echo "Start converting formats.."

src_path='raw_videos'
dst_path='raw_videos_mp4'

if [ ! -f "${dst_path}" ]; then
	mkdir ${dst_path}
fi

total=$(ls ${src_path} | wc -l)
i=0

for src_file in ${src_path}/*
do
    ((i=i+1))
    filename=$(basename -- "$src_file"); extension="${filename##*.}";
    dst_file=${dst_path}/$(echo $(basename "${src_file%.*}").mp4)

    if [ -f "${dst_file}" ]; then
	    echo "${i}/${total}, ${dst_file} exists."
	    continue
    fi

    echo "${i} / ${total}, ${filename}"
    
    if [ ${extension} != "mp4" ];
    then
	    ffmpeg -loglevel panic -i ${src_file} -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" ${dst_file}
    else
	    cp ${src_file} ${dst_file}
    fi
done

echo "Finish converting formats.."
