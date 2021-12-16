#!/usr/bin/env python3

import cv2
import csv
import numpy as np
import os

import rospkg

## Submission by Ujjwal Gupta and Mahdi Ghanei

class ImageProcess:
    def __init__(self):
        # default 
        self.height, self.width = 200, 200
        self.resize1, self.resize2 = 32, 32
        self.knn = None
        self.signs = {0:'empty', 1:'left', 2:'right', 3:'dont_enter', 4:'stop', 5:'goal'}
    
    def getImageDir(self, mode='train'):
        '''Return image directory
        '''
        pkgname = 'gryffindor_final_demo'
        path = rospkg.RosPack().get_path(pkgname)
        # print('path', path)
        # imageDirectory = os.path.join(path, '/include/', pkgname, mode+'_images/')
        imageDirectory = path+ '/include/' + pkgname+'/'+  mode + '_images/'
        # print('imageDirectory',imageDirectory)
        return imageDirectory
    
    def readLabels(self, mode='train'):
        ''' Load training images and labels
        '''
        imageDirectory = self.getImageDir(mode)
        print(imageDirectory)
        with open(imageDirectory + mode + '.txt', 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
        return lines

    def getMaskImage(self, img):
        '''creates a mask of an image
        '''
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m, n, _ = np.shape(img)

        # increase the brightness of the middle region (area of interest)
        hsv_img[m//5:4*m//5,:,2] += 50

        # define range of blue color in HSV
        lower_bgr = np.array([0,80,60])
        upper_bgr = np.array([255,255,255])

        # mask coloured signs
        mask = cv2.inRange(hsv_img, lower_bgr, upper_bgr)
        return mask

    def cropImage(self, pre_processed_img, mask, smart_crop=True):
        '''Crops an image based on mask 
        '''
        # default 
        m, n, _ = np.shape(pre_processed_img)

        if smart_crop:
            y_crop_len = int(np.sum(np.sum(mask/255.0,axis=0) > 1))     # num. of columns that have a pixel activated
            x_crop_len = int(np.sum(np.sum(mask/255.0,axis=1) > 1))     # num. of rows that have a pixel activated
            x_shift = int(x_crop_len/2)
            y_shift = int(y_crop_len/2)
        else:
            # coordinates of the center of the sign
            x_shift = 100
            y_shift = 100


        if (mask==0).all(): # if no sign is there
            x_shift, y_shift = 100, 100                 # in case smart_crop cannot find activated pixels
            x_coord, y_coord = x_shift, y_shift
        else:
            x_coord, y_coord = np.mean(np.where(mask>0),axis=1).astype('int')
            # print(x_coord, y_coord)

        processed_img = cv2.bitwise_and(pre_processed_img, pre_processed_img, mask=mask)
        
        if x_coord - x_shift > 0 and x_coord + x_shift < m and y_coord - y_shift > 0 and y_coord + y_shift < n:
            processed_img = processed_img[x_coord-x_shift:x_coord+x_shift,y_coord-y_shift:y_coord+y_shift]
        else:
            processed_img = cv2.resize(processed_img,(self.height, self.width),interpolation=cv2.INTER_AREA)

        
        processed_img = cv2.resize(processed_img, (self.resize1, self.resize2), interpolation=cv2.INTER_AREA)
        
        return processed_img
    
    def trainClassifier(self):
        '''Trains the KNN calssifier
        '''
        ##############################################
        ### Process training images
        lines = self.readLabels(mode='train')
        imageDirectory = self.getImageDir(mode='train')

        train = np.array([]).reshape(0,3 * self.resize1 * self.resize2)
        for i in range(len(lines)):
            img = cv2.imread(imageDirectory +lines[i][0]+".jpg")
            pre_processed_img = img.copy()

            # get mask
            mask = self.getMaskImage(pre_processed_img)
            # crop image based on mask
            processed_img = self.cropImage(pre_processed_img, mask)
            
            processed_img = processed_img.flatten().reshape(1,-1)/255.0
            train = np.vstack((train,processed_img))

        print(np.shape(train))
        train_data = train.astype(np.float32)

        ##############################################
        ### Train calssifier
        # read in training labels
        train_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])

        ### knn classifier
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
        print('Training done!')

    def classifyImage(self, pre_processed_img, k=9):
        '''Classifies a single image
        '''
        # process test images
        new_mask = self.getMaskImage(pre_processed_img)
        processed_img = self.cropImage(pre_processed_img, new_mask)

        test_img = processed_img.flatten().reshape(1,-1)/255.0
        test_img = test_img.astype(np.float32)

        #classifiy as majority
        ret, results, neighbours, dist = self.knn.findNearest(test_img, k)

        # classify on the basis of weights
        weights = []
        near_neighbours = np.unique(neighbours)
        for j in near_neighbours:
            index = np.where(neighbours == j)
            inv_dist = np.reciprocal(dist[index]+0.1)  # add 0.1 to avoid the cases of divisibility by 0
            weight = np.sum(inv_dist)
            weights.append(weight)
        
        ret = near_neighbours[np.argmax(weights)]
        ret = self.signs[ret]
        
        return ret, neighbours, dist#, processed_img


def main():
    '''Run the classifier on the test set
    '''
    img_process = ImageProcess()
    img_process.trainClassifier()
    ##############################################
    ### Test on the test set
    lines = img_process.readLabels(mode='test')
    imageDirectory = img_process.getImageDir(mode='test')

    correct = 0.0
    confusion_matrix = np.zeros((6,6))
    for i in range(len(lines)):
        original_img = cv2.imread(imageDirectory+lines[i][0]+".jpg")
        ret, neighbours, dist = img_process.classifyImage(original_img, k=9)

        test_label = np.int32(lines[i][1])
        
        if test_label == ret:
            print(str(lines[i][0]) + " Correct, " + str(ret))
            correct += 1
            confusion_matrix[np.int32(ret)][np.int32(ret)] += 1
        else:
            confusion_matrix[test_label][np.int32(ret)] += 1
            
            print(str(lines[i][0]) + " Wrong, " + str(test_label) + " classified as " + str(ret))
            print("\tneighbours: " + str(neighbours))
            print("\tdistances: " + str(dist))

    print("\n\nTotal accuracy: " + str(correct/len(lines)))
    print(confusion_matrix)


if __name__ == '__main__':
	main()



