# 用途
* 个人自用于：
	* 拍摄用于图像分类的样本
	* 训练TensorFlow网络
		* ckpt
		* 初始训练
	* 导出pb文件
	* 用pb文件进行predict

## Notice
* 大部分参数位于config.py
* 拍摄图像和predict在CapAndPred.py
* 制作pb文件在freeze_model.py
* 训练网络在：
	* mobilenet_v2_train.py
	* train_net.py
* 输入输出：
	* model文件夹存放cpkt
	* sample_train用于存放训练样本：
		* 样本结构：
		* model
		* --->stuff1
		* ------>stuff1_1.png
		* ------>stuff1_2.png
		* ------>...
		* --->stuff2
		* ------>stuff2_1.png
		* ------>stuff2_2.png
		* ------>...
* 功能模块：
	* load_image用于制作训练样本队列
	* data_aug用于增广训练样本
	* net用于存放网络模型
	* train_net用于存放训练网络主程序
* 目前网络：
	* alexnet
	* cifarnet
	* inception resnet v2
	* inception v4
	* mobilenet v2
	* resnet v2
	* vgg