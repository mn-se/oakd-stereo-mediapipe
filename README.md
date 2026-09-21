
# OAK-D Stereo Hand 3D

OAK-Dの左右モノクロカメラとMediaPipe Hand Landmarkerを使って、手のランドマークを検出し、左右画像の対応点から3D座標を推定するサンプルです。

## 必要なもの

- OAK-D
- Python 3.13以上
- `uv`
- MediaPipe Hand Landmarkerのモデルファイル（別途ダウンロード）

## セットアップ

```powershell
uv sync
```

### モデルのダウンロード

`hand_landmarker.task` はこのリポジトリでは配布しません。MediaPipe公式のモデル配布先からダウンロードしてください。

- [MediaPipe Hand Landmarker model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

ダウンロードしたファイルを、このリポジトリのプロジェクト直下に `hand_landmarker.task` という名前で配置します。

PowerShellでは次のように実行できます。

```powershell
Invoke-WebRequest `
	-Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
	-OutFile ".\hand_landmarker.task"
```

別の場所に配置する場合は、実行時に `--model` で指定できます。

## 実行

```powershell
uv run python oakd_stereo_mediapipe.py --model .\hand_landmarker.task
```

左右の画像ウィンドウが表示され、両方の画像で手が検出されると、手首と人差し指先端の3D座標がコンソールに表示されます。

`q` キーで終了します。

## ONNXモデルの生成

`.task` に含まれるPalm DetectionとHand LandmarkのTFLiteモデルを、ローカルでONNXへ変換できます。

変換を行うときだけ、追加依存関係をインストールします。

```powershell
uv sync --extra conversion
```

```powershell
uv run python convert_task_to_onnx.py --task .\hand_landmarker.task
```

生成されるファイルは次の2つです。

- `models/hand_detector.onnx`
- `models/hand_landmarks_detector.onnx`

生成されたONNXモデルは`.gitignore`で除外しています。`.task`から派生したモデルを再配布しないため、各自の環境で生成してください。ONNX Runtimeで利用する場合は、Palm Detectionのアンカー復号、NMS、手ROI生成、ランドマーク座標の逆変換も実装する必要があります。

生成後は、ONNX Runtime版を次のように実行できます。

```powershell
uv run python oakd_stereo_onnx.py `
	--detector .\models\hand_detector.onnx `
	--landmarks .\models\hand_landmarks_detector.onnx
```

ONNX版では、MediaPipe Tasksを使わず、ONNX RuntimeでPalm DetectionとHand Landmarkを実行します。左右フレームの同期、ステレオ補正、三角測量はMediaPipe版と同じ考え方です。

## MediaPipeとONNXの同一入力比較

同じOAK-Dの左右フレームをMediaPipe版とONNX版へ入力し、21点の座標差を比較できます。追跡の影響を除くため、両方とも初回検出モードで比較します。

```powershell
uv run python compare_backends.py
```

左右それぞれについて、平均誤差、最大誤差、ランドマークごとの座標と誤差が表示されます。

## 処理構成

- 左右カメラのフレームをシーケンス番号で同期
- 工場出荷時キャリブレーションからステレオ補正マップを生成
- 測距用の補正画像は1280x800で処理
- MediaPipe推論用画像は640x400に縮小
- MediaPipe推論は専用プロセスで実行
- 左右のランドマークを三角測量して3D座標へ変換
- 3D座標に指数移動平均を適用

MediaPipe自身が手の検出後にランドマーク推定用の手領域を内部で生成するため、現在のコードでは外部の手ROI切り出しは行っていません。

## 座標系と単位

三角測量の結果はミリメートルで表示します。OAK-Dの工場キャリブレーションから取得した投影行列と左右カメラ間の外部パラメータを使用します。

## 注意点

- CPU推論のため、環境によって処理速度が変わります。
- 左右画像で同時に手が検出されない場合、3D座標は更新されません。
- 光量不足、強い反射、モーションブラーはランドマークと距離精度に影響します。
- MediaPipeやTensorFlow Liteが出力する警告が表示される場合がありますが、推論を妨げない警告もあります。
- 距離精度を確認するときは、既知の距離に物体を置いて実測値と比較してください。

## 補助スクリプト

- `check_oak.py`: 接続されているOAK-Dの確認
- `check_calib.py`: 工場キャリブレーションの確認
- `check_stereo.py`: MediaPipeを使わない左右カメラ映像の確認
