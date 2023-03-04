#!/bin/bash

#黒い画像を削除する
for i in `seq 1 99`
do
	if [ $i -lt 10 ]; then
		find ./0$i/ -type f -size 1000c -exec rm -f {} \;
	else
		find ./$i/ -type f -size 1000c -exec rm -f {} \;
	fi
done 


#ファイルを一つのディレクトリにまとめる

cmd1="cp ./0$i/* ~/LSGAN/data/"
cmd2="cp ./$i/* ~/LSGAN/data/"

for i in `seq 0 99`
do
	if [ $i -lt 10 ]; then
		eval $cmd1
	else
		eval $cmd2
	fi
done

