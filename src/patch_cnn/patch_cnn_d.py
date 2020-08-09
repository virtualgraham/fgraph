# this is an experiment to combine L2Net with VGG16 to create a patch descriptor optimized for classification

import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, Lambda, Dropout, Dense, Input, Concatenate, ReLU, Cropping2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.applications.vgg16 import preprocess_input

import tensorflow_datasets as tfds
import cv2 as cv
import numpy as np
from tensorflow.keras.utils import to_categorical
import os



# L2/HardNet Architecture (with color)?

la01 = ZeroPadding2D(1, input_shape=(32,32, 3))
la02 = Conv2D(64, kernel_size=(3, 3))
la03 = BatchNormalization(epsilon=0.0001, scale=False, center=False)
la04 = ReLU()

la05 = ZeroPadding2D(1)
la06 = Conv2D(64, kernel_size=(3, 3))
la07 = BatchNormalization(epsilon=0.0001, scale=False, center=False)
la08 = ReLU()

la09 = ZeroPadding2D(1)
la10 = Conv2D(128, kernel_size=(3, 3), strides=2)
la11 = BatchNormalization(epsilon=0.0001, scale=False, center=False)
la12 = ReLU()

la13 = ZeroPadding2D(1)
la14 = Conv2D(128, kernel_size=(3, 3))
la15 = BatchNormalization(epsilon=0.0001, scale=False, center=False)
la16 = ReLU()

la17 = ZeroPadding2D(1)
la18 = Conv2D(256, kernel_size=(3, 3), strides=2)
la19 = BatchNormalization(epsilon=0.0001, scale=False, center=False)
la20 = ReLU()

la21 = ZeroPadding2D(1)
la22 = Conv2D(256, kernel_size=(3, 3))
la23 = BatchNormalization(epsilon=0.0001, scale=False, center=False)
la24 = ReLU()

la25 = Dropout(0.1)
la26 = Conv2D(256, kernel_size=(8, 8))
la27 = BatchNormalization(epsilon=0.0001, scale=False, center=False)


# Inputs are a 7 x 7 grid of 32px x 32px patches 
# That is a 224px 224px image, same as many common CNNs trained on ImageNet

input = Input(shape=(224,224,3))

# ((top_crop, bottom_crop), (left_crop, right_crop))
cr00 = Cropping2D(cropping=((0, 192), (0, 192)))(input)
cr01 = Cropping2D(cropping=((0, 192), (32, 160)))(input)
cr02 = Cropping2D(cropping=((0, 192), (64, 128)))(input)
cr03 = Cropping2D(cropping=((0, 192), (96, 96)))(input)
cr04 = Cropping2D(cropping=((0, 192), (128, 64)))(input)
cr05 = Cropping2D(cropping=((0, 192), (160, 32)))(input)
cr06 = Cropping2D(cropping=((0, 192), (192, 0)))(input)

cr10 = Cropping2D(cropping=((32, 160), (0, 192)))(input)
cr11 = Cropping2D(cropping=((32, 160), (32, 160)))(input)
cr12 = Cropping2D(cropping=((32, 160), (64, 128)))(input)
cr13 = Cropping2D(cropping=((32, 160), (96, 96)))(input)
cr14 = Cropping2D(cropping=((32, 160), (128, 64)))(input)
cr15 = Cropping2D(cropping=((32, 160), (160, 32)))(input)
cr16 = Cropping2D(cropping=((32, 160), (192, 0)))(input)

cr20 = Cropping2D(cropping=((64, 128), (0, 192)))(input)
cr21 = Cropping2D(cropping=((64, 128), (32, 160)))(input)
cr22 = Cropping2D(cropping=((64, 128), (64, 128)))(input)
cr23 = Cropping2D(cropping=((64, 128), (96, 96)))(input)
cr24 = Cropping2D(cropping=((64, 128), (128, 64)))(input)
cr25 = Cropping2D(cropping=((64, 128), (160, 32)))(input)
cr26 = Cropping2D(cropping=((64, 128), (192, 0)))(input)

cr30 = Cropping2D(cropping=((96, 96), (0, 192)))(input)
cr31 = Cropping2D(cropping=((96, 96), (32, 160)))(input)
cr32 = Cropping2D(cropping=((96, 96), (64, 128)))(input)
cr33 = Cropping2D(cropping=((96, 96), (96, 96)))(input)
cr34 = Cropping2D(cropping=((96, 96), (128, 64)))(input)
cr35 = Cropping2D(cropping=((96, 96), (160, 32)))(input)
cr36 = Cropping2D(cropping=((96, 96), (192, 0)))(input)

cr40 = Cropping2D(cropping=((128, 64), (0, 192)))(input)
cr41 = Cropping2D(cropping=((128, 64), (32, 160)))(input)
cr42 = Cropping2D(cropping=((128, 64), (64, 128)))(input)
cr43 = Cropping2D(cropping=((128, 64), (96, 96)))(input)
cr44 = Cropping2D(cropping=((128, 64), (128, 64)))(input)
cr45 = Cropping2D(cropping=((128, 64), (160, 32)))(input)
cr46 = Cropping2D(cropping=((128, 64), (192, 0)))(input)

cr50 = Cropping2D(cropping=((160, 32), (0, 192)))(input)
cr51 = Cropping2D(cropping=((160, 32), (32, 160)))(input)
cr52 = Cropping2D(cropping=((160, 32), (64, 128)))(input)
cr53 = Cropping2D(cropping=((160, 32), (96, 96)))(input)
cr54 = Cropping2D(cropping=((160, 32), (128, 64)))(input)
cr55 = Cropping2D(cropping=((160, 32), (160, 32)))(input)
cr56 = Cropping2D(cropping=((160, 32), (192, 0)))(input)

cr60 = Cropping2D(cropping=((192, 0), (0, 192)))(input)
cr61 = Cropping2D(cropping=((192, 0), (32, 160)))(input)
cr62 = Cropping2D(cropping=((192, 0), (64, 128)))(input)
cr63 = Cropping2D(cropping=((192, 0), (96, 96)))(input)
cr64 = Cropping2D(cropping=((192, 0), (128, 64)))(input)
cr65 = Cropping2D(cropping=((192, 0), (160, 32)))(input)
cr66 = Cropping2D(cropping=((192, 0), (192, 0)))(input)


cs00 = Cropping2D(cropping=((16, 176), (16, 176)))(input)
cs01 = Cropping2D(cropping=((16, 176), (48, 144)))(input)
cs02 = Cropping2D(cropping=((16, 176), (80, 112)))(input)
cs03 = Cropping2D(cropping=((16, 176), (112, 80)))(input)
cs04 = Cropping2D(cropping=((16, 176), (144, 48)))(input)
cs05 = Cropping2D(cropping=((16, 176), (176, 16)))(input)

cs10 = Cropping2D(cropping=((48, 144), (16, 176)))(input)
cs11 = Cropping2D(cropping=((48, 144), (48, 144)))(input)
cs12 = Cropping2D(cropping=((48, 144), (80, 112)))(input)
cs13 = Cropping2D(cropping=((48, 144), (112, 80)))(input)
cs14 = Cropping2D(cropping=((48, 144), (144, 48)))(input)
cs15 = Cropping2D(cropping=((48, 144), (176, 16)))(input)

cs20 = Cropping2D(cropping=((80, 112), (16, 176)))(input)
cs21 = Cropping2D(cropping=((80, 112), (48, 144)))(input)
cs22 = Cropping2D(cropping=((80, 112), (80, 112)))(input)
cs23 = Cropping2D(cropping=((80, 112), (112, 80)))(input)
cs24 = Cropping2D(cropping=((80, 112), (144, 48)))(input)
cs25 = Cropping2D(cropping=((80, 112), (176, 16)))(input)

cs30 = Cropping2D(cropping=((112, 80), (16, 176)))(input)
cs31 = Cropping2D(cropping=((112, 80), (48, 144)))(input)
cs32 = Cropping2D(cropping=((112, 80), (80, 112)))(input)
cs33 = Cropping2D(cropping=((112, 80), (112, 80)))(input)
cs34 = Cropping2D(cropping=((112, 80), (144, 48)))(input)
cs35 = Cropping2D(cropping=((112, 80), (176, 16)))(input)

cs40 = Cropping2D(cropping=((144, 48), (16, 176)))(input)
cs41 = Cropping2D(cropping=((144, 48), (48, 144)))(input)
cs42 = Cropping2D(cropping=((144, 48), (80, 112)))(input)
cs43 = Cropping2D(cropping=((144, 48), (112, 80)))(input)
cs44 = Cropping2D(cropping=((144, 48), (144, 48)))(input)
cs45 = Cropping2D(cropping=((144, 48), (176, 16)))(input)

cs50 = Cropping2D(cropping=((176, 16), (16, 176)))(input)
cs51 = Cropping2D(cropping=((176, 16), (48, 144)))(input)
cs52 = Cropping2D(cropping=((176, 16), (80, 112)))(input)
cs53 = Cropping2D(cropping=((176, 16), (112, 80)))(input)
cs54 = Cropping2D(cropping=((176, 16), (144, 48)))(input)
cs55 = Cropping2D(cropping=((176, 16), (176, 16)))(input)




op00 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr00)))))))))))))))))))))))))))
op01 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr01)))))))))))))))))))))))))))
op02 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr02)))))))))))))))))))))))))))
op03 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr03)))))))))))))))))))))))))))
op04 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr04)))))))))))))))))))))))))))
op05 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr05)))))))))))))))))))))))))))
op06 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr06)))))))))))))))))))))))))))

op10 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr10)))))))))))))))))))))))))))
op11 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr11)))))))))))))))))))))))))))
op12 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr12)))))))))))))))))))))))))))
op13 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr13)))))))))))))))))))))))))))
op14 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr14)))))))))))))))))))))))))))
op15 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr15)))))))))))))))))))))))))))
op16 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr16)))))))))))))))))))))))))))

op20 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr20)))))))))))))))))))))))))))
op21 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr21)))))))))))))))))))))))))))
op22 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr22)))))))))))))))))))))))))))
op23 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr23)))))))))))))))))))))))))))
op24 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr24)))))))))))))))))))))))))))
op25 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr25)))))))))))))))))))))))))))
op26 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr26)))))))))))))))))))))))))))

op30 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr30)))))))))))))))))))))))))))
op31 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr31)))))))))))))))))))))))))))
op32 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr32)))))))))))))))))))))))))))
op33 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr33)))))))))))))))))))))))))))
op34 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr34)))))))))))))))))))))))))))
op35 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr35)))))))))))))))))))))))))))
op36 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr36)))))))))))))))))))))))))))

op40 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr40)))))))))))))))))))))))))))
op41 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr41)))))))))))))))))))))))))))
op42 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr42)))))))))))))))))))))))))))
op43 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr43)))))))))))))))))))))))))))
op44 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr44)))))))))))))))))))))))))))
op45 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr45)))))))))))))))))))))))))))
op46 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr46)))))))))))))))))))))))))))

op50 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr50)))))))))))))))))))))))))))
op51 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr51)))))))))))))))))))))))))))
op52 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr52)))))))))))))))))))))))))))
op53 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr53)))))))))))))))))))))))))))
op54 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr54)))))))))))))))))))))))))))
op55 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr55)))))))))))))))))))))))))))
op56 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr56)))))))))))))))))))))))))))

op60 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr60)))))))))))))))))))))))))))
op61 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr61)))))))))))))))))))))))))))
op62 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr62)))))))))))))))))))))))))))
op63 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr63)))))))))))))))))))))))))))
op64 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr64)))))))))))))))))))))))))))
op65 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr65)))))))))))))))))))))))))))
op66 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cr66)))))))))))))))))))))))))))



oq00 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs00)))))))))))))))))))))))))))
oq01 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs01)))))))))))))))))))))))))))
oq02 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs02)))))))))))))))))))))))))))
oq03 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs03)))))))))))))))))))))))))))
oq04 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs04)))))))))))))))))))))))))))
oq05 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs05)))))))))))))))))))))))))))

oq10 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs10)))))))))))))))))))))))))))
oq11 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs11)))))))))))))))))))))))))))
oq12 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs12)))))))))))))))))))))))))))
oq13 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs13)))))))))))))))))))))))))))
oq14 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs14)))))))))))))))))))))))))))
oq15 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs15)))))))))))))))))))))))))))

oq20 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs20)))))))))))))))))))))))))))
oq21 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs21)))))))))))))))))))))))))))
oq22 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs22)))))))))))))))))))))))))))
oq23 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs23)))))))))))))))))))))))))))
oq24 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs24)))))))))))))))))))))))))))
oq25 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs25)))))))))))))))))))))))))))

oq30 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs30)))))))))))))))))))))))))))
oq31 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs31)))))))))))))))))))))))))))
oq32 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs32)))))))))))))))))))))))))))
oq33 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs33)))))))))))))))))))))))))))
oq34 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs34)))))))))))))))))))))))))))
oq35 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs35)))))))))))))))))))))))))))

oq40 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs40)))))))))))))))))))))))))))
oq41 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs41)))))))))))))))))))))))))))
oq42 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs42)))))))))))))))))))))))))))
oq43 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs43)))))))))))))))))))))))))))
oq44 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs44)))))))))))))))))))))))))))
oq45 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs45)))))))))))))))))))))))))))

oq50 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs50)))))))))))))))))))))))))))
oq51 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs51)))))))))))))))))))))))))))
oq52 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs52)))))))))))))))))))))))))))
oq53 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs53)))))))))))))))))))))))))))
oq54 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs54)))))))))))))))))))))))))))
oq55 = la27(la26(la25(la24(la23(la22(la21(la20(la19(la18(la17(la16(la15(la14(la13(la12(la11(la10(la09(la08(la07(la06(la05(la04(la03(la02(la01(cs55)))))))))))))))))))))))))))



ops =  [op00, op01, op02, op03, op04, op05, op06, 
        op10, op11, op12, op13, op14, op15, op16, 
        op20, op21, op22, op23, op24, op25, op26,
        op30, op31, op32, op33, op34, op35, op36,
        op40, op41, op42, op43, op44, op45, op46,
        op50, op51, op52, op53, op54, op55, op56,
        op60, op61, op62, op63, op64, op65, op66,
        oq00, oq01, oq02, oq03, oq04, oq05,
        oq10, oq11, oq12, oq13, oq14, oq15, 
        oq20, oq21, oq22, oq23, oq24, oq25,
        oq30, oq31, oq32, oq33, oq34, oq35,
        oq40, oq41, oq42, oq43, oq44, oq45, 
        oq50, oq51, oq52, oq53, oq54, oq55]

merged = Concatenate()(ops)


# Fully connected layers from VGG16

x = Dense(4096, activation='relu', name='fc1')(merged)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(1000, activation='softmax', name='predictions')(x)

model = Model(input, x)

model.compile(loss='categorical_crossentropy', optimizer = "adam", metrics=['accuracy'])

print(model.summary())


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))




# Fetch the dataset directly
imagenet = tfds.image.Imagenet2012()
## or by string name
#imagenet = tfds.builder('imagenet2012')

# Describe the dataset with DatasetInfo
C = imagenet.info.features['label'].num_classes
Ntrain = imagenet.info.splits['train'].num_examples
Nvalidation = imagenet.info.splits['validation'].num_examples
Nbatch = 32
assert C == 1000
assert Ntrain == 1281167
assert Nvalidation == 50000

# Download the data, prepare it, and write it to disk
imagenet.download_and_prepare()

# Load data from disk as tf.data.Datasets
datasets = imagenet.as_dataset()
train_dataset, validation_dataset = datasets['train'], datasets['validation']

# print("type(train_dataset).__name__",type(train_dataset).__name__)
# print("type(validation_dataset).__name__", type(validation_dataset).__name__)
# assert isinstance(train_dataset, tf.data.PrefetchDataset)
# assert isinstance(validation_dataset, tf.data.PrefetchDataset)

def imagenet_generator(dataset, batch_size=32, num_classes=1000, is_training=False):
  images = np.zeros((batch_size, 224, 224, 3))
  labels = np.zeros((batch_size, num_classes))
  while True:
    count = 0 
    for sample in tfds.as_numpy(dataset):
      image = sample["image"]
      label = sample["label"]
    
      images[count%batch_size] = preprocess_input(np.expand_dims(cv.resize(image, (224, 224)), 0))
      labels[count%batch_size] = np.expand_dims(to_categorical(label, num_classes=num_classes), 0)
      
      count += 1
      if (count%batch_size == 0):
        yield images, labels

# Infer on ImageNet
# labels = np.zeros((Nvalidation))
# pred_labels = np.zeros((Nvalidation, C))
# pred_labels_new = np.zeros((Nvalidation, C))
  
# score = model.evaluate_generator(imagenet_generator(validation_dataset,batch_size=32), 
#                                   steps= Nvalidation // Nbatch, 
#                                   verbose=1)
# print("Evaluation Result of Original Model on ImageNet2012: " + str(score))
  
# Train on ImageNet
checkpoint_path = "patchnet/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
os.makedirs(checkpoint_dir, exist_ok=True)

cp_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 1-epoch
    period=1)
    
csv_logger = keras.callbacks.CSVLogger('patchnet_training.csv')
  
print("Starting to train patchnet...")
epochs = 20

model.fit_generator(imagenet_generator(train_dataset, batch_size=Nbatch, is_training=True),
                        steps_per_epoch= Ntrain // Nbatch,
                        epochs = epochs,
                        validation_data = imagenet_generator(validation_dataset, batch_size=Nbatch),
                        validation_steps = Nvalidation // Nbatch,
                        verbose = 1,
                        callbacks = [cp_callback, csv_logger])
  
model.save("patchnet.h5")