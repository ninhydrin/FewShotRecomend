# FewShotRecommend
    
## Requirements
- Python 2.7+
- NumPy
- Pickle
- Urllib
- Tqdm
- OpenCV
- Librosa
- Skimage
- Pylab
- (scikit.samplerate)
- Chainer (1.5.1): https://github.com/pfnet/chainer

 Anaconda-2.3.0 の使用を推奨
 
## How to use
positiveにお気に入りの音楽を、negativeに嫌い（というか趣向に合わない）音楽を入れる（.wavにのみ対応。100曲未満ずつが理想）。
その後FewShotRecommendディレクトリ内で

./weakRecommender.sh train

を実行し好みを学習（PCのスペックによっては時間がかかる）。
学習終了後は音楽ファイルの入ったディレクトリを指定して

./weakRecommender.sh predict path/to/music_dir

resultディレクトリに結果が保存される。

あくまでも予測値です
## Other
gpuの使用を推奨

(Mac OS X Yosemite 10.10.3にて動作確認)

./weakRecommender.sh itunes

でiTunesのデータを学習用サンプルとして取得する。再生回数が中央値以上ならpositive、中央値未満ならnegativeに分類される。環境によってはうまくいかないので、引数にiTunesディレクトリを指定する
