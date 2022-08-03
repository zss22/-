# -
import pandas as pd
import numpy as np

%pylab inline
import seaborn as sns

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df['Activity'] = train_df['Activity'].map({
    'LAYING': 0,
    'STANDING': 1,
    'SITTING': 2,
    'WALKING': 3,
    'WALKING_UPSTAIRS': 4,
    'WALKING_DOWNSTAIRS': 5
})
data=pd.concat([train_df,test_df])
X1=data.drop(columns="Activity")
import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=0.9)   #保留90%的数据
pca.fit(X1)                  #训练
X_r=pca.fit_transform(X1)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)
#print(pca.explained_variance_ratio_)  #输出贡献率
X_r=pd.DataFrame(X_r)
train_df2=pd.DataFrame(X_r[0:8000])
test_df2=pd.DataFrame(X_r[8000:10000])
#搭建模型
import sklearn.svm as svm
from sklearn.model_selection import KFold,cross_val_score as CVS,train_test_split as TTS
from sklearn.metrics import accuracy_score

train_X = train_df2.iloc[:,0:561] #提取特征集
train_Y = train_df['Activity'] #提取标签
test_X = test_df2.iloc[:, :]
xtrain,xtest,ytrain,ytest = TTS(train_X,train_Y,test_size = 0.09, random_state = 200)
r = []
def modle_pro(modle_):
    ## 创建保存模型信息的list
    modle_r_list = []
    ## 初始化模型
    modle = modle_
    ## 记录模型训练运行时间
    old_time = time.time()
    modle.fit(xtrain,ytrain)
    current_time = time.time()
    use_time = round(current_time-old_time,4)
    ## 模型预测
    pre = modle.predict(xtest)
    pre = pre.astype(np.int64)
    ## 由于其问题本质属于分类问题，故对模型评估统一采用模型预测准确率(accuracy_score)进行评估
    acc_score = round(accuracy_score(pre,ytest),4)
    ##
    modle_r_list.append(str(modle_))
    modle_r_list.append(use_time)
    modle_r_list.append(acc_score)
    r.append(modle_r_list)
    return acc_score
    #SVM
from sklearn.svm import SVC
svm = SVC(kernel = 'rbf',C=4.2,tol=0.001,gamma=0.1)
modle_pro(svm)
#用于提交的测试结果
test_predict = svm.predict(test_X)
test_predict = pd.DataFrame({'Activity': test_predict})
test_predict['Activity'] = test_predict['Activity'].map({
    0:'LAYING',
    1:'STANDING',
    2:'SITTING',
    3:'WALKING',
    4:'WALKING_UPSTAIRS',
    5:'WALKING_DOWNSTAIRS'
})

test_predict.to_csv('submission.csv', index=None)
