这个项目来自:https://github.com/ultralytics/yolov3
不喜欢他的数据读取方式, 做了一点修改, 对代码也有一些修改, 但基本没什么影响

使用方式:
1. git clone https://github.com/TWSFar/YOLO-V3.git

2. 修改train.py的classes

3. 将--root_path设置为数据集的根目录, 数据集应该是如下形式:
    root:
        JPEGImages
        ImageSets/Main/train.txt val.txt
        Annotations

4. 修改--weights为预训练权重地址, 为空的将会自动下载

5. python train.py


