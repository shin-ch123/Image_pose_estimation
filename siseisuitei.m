%% ネットワークのインポート

%ONNX ファイルから事前学習済みのネットワークをインポート
dataDir = fullfile(tempdir,'OpenPose');
trainedOpenPoseNet_url = 'https://ssd.mathworks.com/supportfiles/vision/data/human-pose-estimation.zip';
downloadTrainedOpenPoseNet(trainedOpenPoseNet_url,dataDir)
%zipファイルの展開
unzip(fullfile(dataDir,'human-pose-estimation.zip'),dataDir);
%LayerGraph objectの作成
modelfile = fullfile(dataDir,'human-pose-estimation.onnx');
layers = importONNXLayers(modelfile,"ImportWeights",true);
%未使用の出力レイヤーを削除
layers = removeLayers(layers,layers.OutputNames);
net = dlnetwork(layers);

%% テスト イメージのヒートマップと PAF の予測
im = imread("sakana--.jpg");
%画像の再スケール化
netInput = im2single(im)-0.5;
netInput = netInput(:,:,[3 2 1]);
netInput = dlarray(netInput,"SSC");
[heatmaps,pafs] = predict(net,netInput);
heatmaps = extractdata(heatmaps);
montage(rescale(heatmaps),"BackgroundColor","b","BorderSize",3)
%ブライトスポットとボディの対応を視覚化するために、最初のヒートマップをテスト画像の上にfalsecolorで表示
idx = 1;
hmap = heatmaps(:,:,idx);
hmap = imresize(hmap,size(im,[1 2]));
figure
imshowpair(hmap,im);
%背景のヒートマップを使用
heatmaps = heatmaps(:,:,1:end-1);
%dlarrayに格納されているPAFデータを取得
pafs = extractdata(pafs);
montage(rescale(pafs),"Size",[19 2],"BackgroundColor","b","BorderSize",3)
%PAFと身体の対応関係を視覚化するために、テスト画像上に第一種身体部位ペアのx成分とy成分をfalsecolorで表示
idx = 1;
impair = horzcat(im,im);
pafpair = horzcat(pafs(:,:,2*idx-1),pafs(:,:,2*idx));
pafpair = imresize(pafpair,size(impair,[1 2]));
imshowpair(pafpair,impair);
%% ヒートマップとPAFからポーズを特定する
params = getBodyPoseParameters;
%3次元の行列を返す
poses = getBodyPoses(heatmaps,pafs,params);
%renderBodyPosesヘルパー関数を使用し、ボディポーズを表示
renderBodyPoses(im,poses,size(heatmaps,1),size(heatmaps,2),params);