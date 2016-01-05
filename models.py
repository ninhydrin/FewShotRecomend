import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers

class Chroma(chainer.Chain):
    insize = 12
    def __init__(self,pre=False):
        category_num = 10 if pre else 2
        super(Chroma, self).__init__(
            fc1=L.Linear(12, 128),
            #fc2=L.Linear(256, 128),
            fc3=L.Linear(128, 128),
            fc_last=L.Linear(128, category_num)
        )
        self.train = True

    def __call__(self, x, t):
        h = F.leaky_relu(self.fc1(x),slope=0.3)
        #h = F.dropout(F.relu(self.fc1(h)), train=self.train,ratio = .7)  
        #h = F.dropout(F.relu(self.fc2(h)), train=self.train,ratio = .7)  
        h = F.dropout(F.relu(self.fc3(h)), train=self.train,ratio = .7)  
        h = self.fc_last(h)
        self.pre = F.softmax(h)
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss

class Mel(chainer.Chain):
    insize = 128
    def __init__(self,pre=False):
        category_num = 10 if pre else 2
        super(Mel, self).__init__(
            #conv1=F.Convolution2D(1,23,ksize=8,stride=4),
            #conv2=F.Convolution2D(23,10,ksize=4),
            #conv3=F.Convolution2D(10,6,ksize=3),
            #conv4=F.Convolution2D(6,6,ksize=3),
            fc1=L.Linear(128,1024),
            fc3=L.Linear(1024, 1024),
            fc4=L.Linear(1024, 512),
            fc_last=L.Linear(512, category_num),
        )
        self.train = True

    def __call__(self, x, t):
        #h = F.relu(self.conv1(x))
        #h = F.relu(self.conv2(h))
        #h = F.relu(self.conv3(h))
        #h = F.relu(self.conv4(h))
        h = F.dropout(F.leaky_relu(self.fc1(x),slope=0.1), train=self.train)        
        #h = F.dropout(F.relu(self.fc2(h)), train=self.train)
        h = F.dropout(F.relu(self.fc3(h)), train=self.train,ratio=.7)
        h = F.dropout(F.relu(self.fc4(h)), train=self.train,ratio=.7)
        h = self.fc_last(h)
        self.pre = F.softmax(h)
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss


class Mfcc(chainer.Chain):

    insize = 120
    def __init__(self,pre=False):
        category_num = 10 if pre else 2
        super(Mfcc, self).__init__(
            #conv1=F.Convolution2D(1,24,ksize=(10,1),stride=5),
            #conv2=F.Convolution2D(24,12,ksize=4),
            #conv3=F.Convolution2D(12,8,ksize=3),
            #conv4=F.Convolution2D(8,8,ksize=3),
            fc1=L.Linear(120, 512),            #fc2=L.Linear(512, 1024),
            fc2=L.Linear(512, 1024),
            fc3=L.Linear(1024, 1024),
            fc4=L.Linear(1024, 512),
            fc5=L.Linear(512, 512),
            fc_last=L.Linear(512, category_num)
        )
        self.train = True

    def __call__(self, x, t):
        #h = F.relu(self.conv1(x))
        #h = F.relu(self.conv2(h))
        #h = F.relu(self.conv3(h))
        #h = F.relu(self.conv4(h))
        h = F.dropout(F.leaky_relu(self.fc1(x),slope=0.05), train=self.train,ratio=.8)        
        h = F.dropout(F.sigmoid(self.fc2(h)), train=self.train,ratio=.7)
        h = F.dropout(F.relu(self.fc3(h)), train=self.train,ratio =.7)
        h = F.dropout(F.leaky_relu(self.fc4(h),slope=0.3), train=self.train,ratio=.7)
        h = F.dropout(F.leaky_relu(self.fc5(h),slope=0.03), train=self.train)
        h = self.fc_last(h)
        self.pre = F.softmax(h)
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss

class Law(chainer.Chain):
    insize = 5000
    def __init__(self,pre=False):
        category_num = 10 if pre else 2
        super(Law, self).__init__(
            conv1=F.Convolution2D(1,24,ksize=(1,100),stride=5),
            #conv2=F.Convolution2D(24,12,ksize=(1,6)),
            conv2=F.Convolution2D(24,12,ksize=(1,20)),
            conv3=F.Convolution2D(12,8,ksize=(1,4)),            
            #conv3=F.Convolution2D(12,8,ksize=3),
            conv4=F.Convolution2D(8,8,ksize=(1,4)),
            #fc1=L.Linear(120, 512),            #fc2=L.Linear(512, 1024),
            #fc2=L.Linear(120*98, 1024),
            #fc3=L.Linear(3856, 2048),
            #fc3=L.Linear(3848, 2048),
            #fc4=L.Linear(2048, 512),
            #fc4=L.Linear(3848, 512),
            #fc4=L.Linear(3736, 2048),
            fc4=L.Linear(3720, 2048),
            fc6=L.Linear(2048, 512),
            fc5=L.Linear(512, 512),
            fc_last=L.Linear(512, category_num)
        )
        self.train = True

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.leaky_relu(self.conv1(x),slope=0.1),(1,4),stride=2)
        h = F.leaky_relu(self.conv2(h),slope=0.3)
        h = F.leaky_relu(self.conv3(h),slope=0.6)
        h = F.leaky_relu(self.conv4(h),slope=0.3)
        #h = F.dropout(F.relu(self.fc1(x)), train=self.train)        
        #h = F.dropout(F.relu(self.fc2(x)), train=self.train,ratio=.7)
        #h = F.dropout(F.relu(self.fc3(h)), train=self.train)
        h = F.dropout(F.relu(self.fc4(h)), train=self.train,ratio=.7)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train,ratio=.6)
        h = F.dropout(F.relu(self.fc5(h)), train=self.train,ratio=.6)
        h = self.fc_last(h)
        
        self.pre = F.softmax(h)
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss

class Spectro(chainer.Chain):

    insize = 256

    def __init__(self,pre=False):
        category_num = 10 if pre else 2
        super(Spectro, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
            conv2=L.Convolution2D(96, 256,  5, pad=2),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3),
            fc6=L.Linear(9216, 4096),
            #fc6=L.Linear(28800, 4096),
            fc7=L.Linear(4096, 4096),
            fc_last=L.Linear(4096, category_num),
        )
        self.train = True

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        #h = F.relu(self.conv6(h))
        
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        self.hidden_feature = h
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc_last(h)
        
        self.pre = F.softmax(h)
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss
                    
class DivideConv(chainer.Chain):
    insize = 256
    def __init__(self):
        super(DivideConv, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
            conv2=L.Convolution2D(96, 256,  5, pad=2),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3),
        )
