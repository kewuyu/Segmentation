import numpy as np


class Metrics(object):
    def __init__(self, num_classes):
        '''

        :param num_classes(int): 分类数
        '''
        self.numclas = num_classes
        self.confusionMatrix = np.zeros((self.numclas,) * 2)
        #创建一个 num_classes * num_classes的全零矩阵

    def PA(self):
        return np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        # PA = acc = (TP+TN) / (TP + TN + FP + FN)
        # 对角线数值的 / 混淆矩阵所有数值的和

    def CPA(self):
        return np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        # 某一类别的acc
        # 某一类别预测正确的个数 / 预测为某一类的数值；对角线的值 / 对应列数的和

    def mCPA(self):
        classAcc = self.cpa()
        meanACC = np.nanmean(classAcc)
        return meanACC
        # 所有类别cpa的平均值
    def mIou(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def genConfusionMatrix(self, labels, hat_labels):
        mask = (labels >= 0) & (labels < self.numclas)
        hist = np.bincount(self.numclas * labels[mask] + hat_labels[mask], minlength=self.numclas ** 2).reshape(
            self.numclas, self.numclas)
        return hist

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addbatch(self, labels, hat_labels):
        assert hat_labels.shape == labels.shape
        self.confusionMatrix += self.genConfusionMatrix(labels, hat_labels)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numclas, self.numclas))
