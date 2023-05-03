wget https://github.com/jianhao2016/AllSet/raw/main/data/raw_data/AllSet_all_raw_data.zip
unzip AllSet_all_raw_data.zip
if [ ! -d "mtx" ]
then
    mkdir mtx
fi
if [ ! -d "lap_mtx" ]
then
    mkdir lap_mtx
fi
python ../prepare_data.py
rm -r AllSet_all_raw_data
rm AllSet_all_raw_data.zip