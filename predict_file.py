import os

import keras
import numpy as np
from PIL import Image
from PIL import ImageColor
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


def print_comparison_between_expectation_and_prediction_images(to_green, to_orange, to_red, width, height, img_path,
                                                               comparision_index, model_nr, prag):
    my_radiography = Image.open(img_path, 'r')
    comparison_image = Image.new('RGBA', (width, height))
    comparison_image.paste(my_radiography)
    for pair in to_green:
        comparison_image.putpixel((pair[0], pair[1]), ImageColor.getcolor('green', 'RGBA'))
    for pair in to_red:
        comparison_image.putpixel((pair[0], pair[1]), ImageColor.getcolor('red', 'RGBA'))
    for pair in to_orange:
        comparison_image.putpixel((pair[0], pair[1]), ImageColor.getcolor('orange', 'RGBA'))
    comparison_image.save('resultscupragcuboala{}\{}.png'.format(prag, comparision_index))

    true_positives = len(to_green)
    true_negatives = width * width - len(to_green) - len(to_orange) - len(to_red)
    false_positives = len(to_orange)
    false_negatives = len(to_red)
    sensitivity = true_positives / width * width

    return true_positives, true_negatives, false_positives, false_negatives, sensitivity


def compare_image_arrays(width, height, expectation_image, prediction_image, img_path, comparision_index, model_nr,
                         prag):
    to_red = []
    to_green = []
    to_orange = []
    expectation_image = expectation_image.transpose()
    prediction_image = prediction_image.transpose()
    exist_ill_in_img_pred = False
    exist_ill_in_img_test = False
    for row in range(height):
        for col in range(width):
            if expectation_image[row][col] != 0 and prediction_image[row][col] != 0.0:
                to_green.append((row, col))
                expectation_image[row][col] = 1
                prediction_image[row][col] = 1
                exist_ill_in_img_pred = True
                exist_ill_in_img_test = True

            elif expectation_image[row][col] != 0 and prediction_image[row][col] == 0.0:
                to_orange.append((row, col))
                expectation_image[row][col] = 1
                prediction_image[row][col] = 0
                exist_ill_in_img_test = True

            elif expectation_image[row][col] == 0 and prediction_image[row][col] != 0.0:
                to_red.append((row, col))
                expectation_image[row][col] = 0
                prediction_image[row][col] = 1
                exist_ill_in_img_pred = True

            else:
                expectation_image[row][col] = 0
                prediction_image[row][col] = 0
    print_comparison_between_expectation_and_prediction_images(to_green, to_orange, to_red,
                                                               width, height, img_path,
                                                               comparision_index, model_nr, prag)
    expectation_image = expectation_image.reshape(width * width, 1)
    prediction_image = prediction_image.reshape(width * width, 1)
    accuracy = accuracy_score(expectation_image, prediction_image)
    recall = recall_score(expectation_image, prediction_image, zero_division=0, average="micro")
    precision = precision_score(expectation_image, prediction_image, zero_division=0, average="micro")
    f1 = f1_score(expectation_image, prediction_image, zero_division=0)

    if len(to_green) + len(to_red) + len(to_orange) == 0.0:
        iou = 1.0
        dice_coef = 1.0
    elif len(to_green) == 0:
        iou = 0.0
        dice_coef = 2 * len(to_green) / (2 * len(to_green) + len(to_red) + len(to_orange))
    else:
        dice_coef = 2 * len(to_green) / (2 * len(to_green) + len(to_red) + len(to_orange))
        iou = len(to_green) / (len(to_green) + len(to_red) + len(to_orange))
        # print(iou)
    return accuracy, recall, precision, f1, iou, dice_coef, exist_ill_in_img_test, exist_ill_in_img_pred


def prepare_dataset(my_path):
    radiographies_pixels = []
    ok = 0
    for my_file in os.listdir(my_path):
        # print(os.path.join(my_path, my_file))
        ok += 1
        if ok > 1000:
            break
        my_radiography = Image.open(os.path.join(my_path, my_file), 'r')
        radiography = list(my_radiography.getdata())
        radiography = np.array(radiography)
        radiographies_pixels.append(radiography)
    radiographies_pixels = np.array(radiographies_pixels, dtype='uint8')
    return radiographies_pixels


second_list_models = [r'C:\Users\Lucia\PycharmProjects\untitled5\secondweights.100-174.41.h5',
                      r'C:\Users\Lucia\PycharmProjects\untitled5\secondweights.200-174.39.h5',
                      r'C:\Users\Lucia\PycharmProjects\untitled5\secondweights.300-174.39.h5',
                      r'C:\Users\Lucia\PycharmProjects\untitled5\secondweights.400-174.39.h5',
                      r'C:\Users\Lucia\PycharmProjects\untitled5\secondweights.500-174.38.h5',
                      r'C:\Users\Lucia\PycharmProjects\untitled5\secondweights.600-174.38.h5',
                      r'C:\Users\Lucia\PycharmProjects\untitled5\secondweights.700-174.38.h5',
                      r'C:\Users\Lucia\PycharmProjects\untitled5\secondweights.800-174.38.h5',
                      r'C:\Users\Lucia\PycharmProjects\untitled5\secondweights.900-174.38.h5']

list_models = [r'C:\Users\Lucia\PycharmProjects\untitled5\weights.100-172.90.h5',
               r'C:\Users\Lucia\PycharmProjects\untitled5\weights.200-172.87.h5',
               r'C:\Users\Lucia\PycharmProjects\untitled5\weights.300-172.86.h5',
               r'C:\Users\Lucia\PycharmProjects\untitled5\weights.400-172.86.h5',
               r'C:\Users\Lucia\PycharmProjects\untitled5\weights.500-172.86.h5',
               r'C:\Users\Lucia\PycharmProjects\untitled5\weights.600-172.86.h5',
               r'C:\Users\Lucia\PycharmProjects\untitled5\weights.700-172.85.h5',
               r'C:\Users\Lucia\PycharmProjects\untitled5\weights.800-172.86.h5',
               r'C:\Users\Lucia\PycharmProjects\untitled5\weights.900-172.85.h5',
               r'C:\Users\Lucia\PycharmProjects\untitled5\weights.1000-172.85.h5']
l_models = [r'C:\Users\Lucia\PycharmProjects\untitled5\weights.700-172.85.h5']


def get_prediction_evolution(width, dim_dataset, list_models, test_dicom_path, test_masks_path, prag):
    x_test = prepare_dataset(test_dicom_path)
    y_test = prepare_dataset(test_masks_path)
    x_test = x_test.reshape(dim_dataset, width, width, 1)
    y_test = y_test.reshape(dim_dataset, width, width, 1)
    empty_mask = []
    list_dice_coef = []
    list_clasif_corect = []
    for i in range(width):
        my_list = []
        for j in range(width):
            my_list.append(0)
        empty_mask.append(my_list)

    empty_mask = np.array(empty_mask)
    empty_mask = empty_mask.reshape(width, width, 1)

    for poz, model_name in enumerate(list_models):
        pix_pred_list = []
        my_model = keras.models.load_model(model_name, compile=False)
        for j, y in enumerate(x_test):
            y = np.array(y)
            y = y.reshape(width, width)
            img = Image.fromarray((y).astype(np.uint8))
            img.save('resultscupragcuboala{}\{}.png'.format(prag, j))
        y_pred = my_model.predict(x_test)
        for i, y in enumerate(y_pred):
            nr_pixels_pred = 0
            for j in range(0, len(y)):
                for k in range(0, len(y)):

                    if y[j][k] >= prag:
                        y[j][k] = 255
                        nr_pixels_pred += 1

                    else:
                        y[j][k] = 0
            pix_pred_list.append(nr_pixels_pred)
            y = y.reshape(width, width)

            img = Image.fromarray((y).astype(np.uint8))
            img.save('resultscupragcuboala\{}.png'.format(i))
            # if nr_pixels_pred < prag_min and nr_pixels_pred != 0:
            #     # print(i , nr_pixels_pred)
            #     y_pred[i] = empty_mask

        sum_accuracy = 0
        sum_recall = 0
        sum_f1 = 0
        sum_precision = 0
        sum_iou = 0
        sum_clasif_corect = 0
        sum_dice_coef = 0
        for i in range(0, len(y_pred)):
            expect = np.reshape(y_test[i], (width, width))
            actual = np.reshape(y_pred[i], (width, width))

            accuracy, recall, precision, f1, iou, dice_coef, exist_ill_in_test, exist_ill_in_pred = compare_image_arrays(
                width,
                width,
                expect,
                actual,
                'resultscupragcuboala{}\{}.png'.format(prag, i),
                i, poz + 1, prag)
            # if (pix_pred_list[i] < prag_min):
            #     exist_ill_in_pred = False

            if exist_ill_in_pred == exist_ill_in_test:
                sum_clasif_corect += 1
            sum_dice_coef += dice_coef
            sum_accuracy += accuracy
            sum_recall += recall
            sum_precision += precision
            sum_f1 += f1
            sum_iou += iou
        list_dice_coef.append(sum_dice_coef / dim_dataset)
        list_clasif_corect.append(sum_clasif_corect / dim_dataset)
        print(list_dice_coef)
        print(list_clasif_corect)
    print(list_dice_coef)
    print(list_clasif_corect)
    return list_dice_coef, list_clasif_corect


list_dice_coef, list_clasif_corect = get_prediction_evolution(32, 1000, l_models,
                                                              r'C:\Users\Lucia\Desktop\mydataset\dicom',
                                                              r'C:\Users\Lucia\Desktop\mydataset\mask',
                                                              0.7)

import matplotlib.pyplot as plt


def evolutie_model_cu_prag_minim_pe_pixel(list_dice_coef, list_clasif_corect):
    prag_minim = []
    for i in range(1, 11):
        prag_minim.append(i)

    plt.plot(list_dice_coef, color='orange', label='Dice Coefficient')
    plt.plot(list_clasif_corect, color='purple', label="Acuratețe")
    plt.ylabel('Procent')
    plt.xlabel('Prag')
    plt.xticks(prag_minim)
    plt.savefig('acuratete_si_dice_coef.png')
    plt.legend()
    plt.show()


def evolutie_metrici_cu_nr_minim_de_pixeli_plot(list_dice_coef, list_clasif_corect):
    prag_minim = []

    for i in range(0, 40, 2):
        prag_minim.append(i)

    plt.plot(list_dice_coef, color='orange', label='Dice Coefficient')
    plt.plot(list_clasif_corect, color='purple', label="Acuratețe")
    plt.ylabel('Procent')
    plt.xlabel('Prag')
    plt.xticks(prag_minim)
    plt.legend()
    plt.show()


def evolutie_metrici_pe_cele_zece_modele(list_dice_coef, list_clasif_corect):
    nr_iteratii = []
    for i in range(1, 12):
        nr_iteratii.append(i)
    plt.title('Evoluție modele')
    plt.plot(list_dice_coef, color='orange', label='Dice Coefficient')
    plt.plot(list_clasif_corect, color='purple', label="Acuratețe")
    plt.ylabel('Procent')
    plt.xlabel('Numărul de iterații pentru care a fost antrenat modelul')
    plt.xticks(nr_iteratii)
    plt.savefig('evolutie_model.png')
    plt.legend()
    plt.show()
