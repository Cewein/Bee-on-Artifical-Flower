# download and unzip the dataset
$dir = "./baf/"
if (!(Test-Path $dir)) {
    $url = "https://app.roboflow.com/ds/4TadDrWzxR?key=YBmMXGBZ62"
    $file = "BAF.yolov7pytorch.zip"

    Write-Host "Downloading $file"
    Invoke-WebRequest $url -OutFile $file
    Expand-Archive -Path $file -DestinationPath $dir
    Remove-Item $file
} else {
    Write-Host "Directory $dir exists. Skipping database download."
}

# load submodule, i.e yolov7
git submodule init
git submodule update

# go into the yolo repo for training
cd yolov7

$p = "p5"

# Custom Yolo
$cy = "../training/tl-training.yaml"
$cyw6 = "../training/yolov7-w6-custom.yaml"
$cydf = "../training/yolov7-custom.yaml"
$cyhyp = "../training/hyp.scratch.custom.yaml"

# choose between two types of training
if ($p -eq "p5") {
    $weights = "yolov7_training.pt"
    $img_size = "640 640"
    $name = "yolov7-custom"
} else {
    $weights = "yolov7-w6_training.pt"
    $img_size = "1280 1280"
    $name = "yolov7-w6-custom"
}

$cmd = "train.py" if ($p -eq "p5") else "train_aux.py"
$params = "--workers 4 --device 0 --batch-size 16 --data $cy --img $img_size --cfg $cydf --weights '$weights' --name $name --hyp $cyhyp"

python $cmd $params
