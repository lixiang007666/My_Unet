#参考了Retina_Unet项目，决定自己用代码来实现一遍，数据增强不是像Retina_Unet那样随机裁剪，而是将20个训练数据集按顺序裁剪，每张裁剪成48x48大小的144个patch，20张一共裁剪出2880个patch。
# BY lixiang

#Unet模型：模型输入的张量形状为（Npatch,1,48,48），输出为（Npatch,2304,2）。Npatch表示训练集的样本数，本例中训练时为2880，预测时为144。
#训练：把原作者代码中的SGD改为Adam，效果有提升。
#预测：也需要先把待预测图像分割成48×48的小图，输入模型，然后把结果整理还原为完整图像，再和专家标注结果进行对比。代码中以测试集第一张图片为例，可自行修改为其他眼底图片路径。

from util import *
from unet import *

if __name__ == '__main__':

    #参数和路径
    resize_height, resize_width = (576, 576)
    dx = 48
    img_path = 'DRIVE/training/images/'
    label_path = 'DRIVE/training/1st_manual/'

    #读取数据并resize
    imglst,images = read_image_and_name(img_path)
    labellst,labels = read_label_and_name(label_path)
    imgs_resize = resize(images,resize_height, resize_width)
    labels_resize = resize(labels,resize_height, resize_width)

    #将imgs列表和manuals列表转换成numpy数组
    X_train = np.array(imgs_resize)
    Y_train = np.array(labels_resize)
    print(X_train.shape)
    print(Y_train.shape)

    #标准化
    X_train = X_train.astype('float32')/255
    Y_train = Y_train.astype('float32')/255

    #提取训练集的G通道
    X_train = X_train[...,1]

    #对训练数据进行裁剪
    X_train = crop(X_train,dx)
    Y_train = crop(Y_train,dx)
    print('X_train shape: '+str(X_train.shape)) #X_train(2880,48,48)
    print('Y_train shape: '+str(Y_train.shape)) #Y_train(2880,48,48)

    #X_train增加一维变成(2880,1,48,48)
    X_train = X_train[:,np.newaxis, ...]
    print('X_train shape: '+str(X_train.shape))
    #Y_train改变shape变成(2880,2304),保持第一维不变，其他维合并
    Y_train = Y_train.reshape(Y_train.shape[0],-1)
    print('Y_train shape: '+str(Y_train.shape))
    Y_train =Y_train[..., np.newaxis]  #增加一维变成(2880,2304,1)
    print('Y_train shape: '+str(Y_train.shape))
    temp = 1 - Y_train
    Y_train = np.concatenate([Y_train, temp], axis=2) #变成(2880,2304,2)
    print('Y_train shape: '+str(Y_train.shape))

    #获得model
    model = get_unet(X_train.shape[1],X_train.shape[2],X_train.shape[3])
    model.summary() #输出参数Param计算过程
    checkpointer = ModelCheckpoint(filepath='best_weights.h5',verbose=1,monitor='val_acc',mode='auto',save_best_only=True)
    model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    #optimizer: 优化器 把原作者代码中的SGD改为Adam，效果有提升。
    model.fit(X_train,Y_train,batch_size=64,epochs=20,verbose=2,shuffle=True,validation_split=0.2,callbacks=[checkpointer])
    print('ok')
