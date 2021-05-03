import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import math
import easyocr

from skimage.filters import threshold_local
from PIL import Image


def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


def plot_rgb(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def plot_gray(image):
    plt.figure(figsize=(16,10))
    plt.imshow(image, cmap='Greys_r')
    plt.show()


def save_gray(image, save_name):
    plt.figure(figsize=(16,10))
    save_path = 'C:/Users/krish/Desktop/CeadarAssignmentCode/ceadar-data-scientist/output_images/' + save_name
    print(save_path)
    plt.imsave(save_path, image, format='jpg', cmap='Greys_r')
    plt.close()


# approximate the contour by a more primitive polygon shape
def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.1 * peri, True)


def get_receipt_contour(contours):    
    # loop over the contours
    for c in contours:
        approx = approximate_contour(c)
        # if our approximated contour has four points, we can assume it is receipt's rectangle
        if len(approx) == 4:
            return approx


def contour_to_rect(contour, resize_ratio):
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    # top-left point has the smallest sum
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points:
    # the top-right will have the minumum difference 
    # the bottom-left will have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect / resize_ratio

def wrap_perspective(img, rect):
    # unpack rectangle points: top left, top right, bottom right, bottom left
    (tl, tr, br, bl) = rect
    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    # destination points which will be used to map the screen to a "scanned" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    # warp the perspective to grab the screen
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))


def bw_scanner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    return (gray > T).astype("uint8") * 255

def preprocessing_adaptive(file_name):
    # Sample file out of the dataset
    img = Image.open(file_name)
    img.thumbnail((800,800), Image.ANTIALIAS)
    
    image = cv2.imread(file_name)
    # Downscale image as finding receipt contour is more efficient on a small image
    resize_ratio = 1000 / image.shape[0]
    original = image.copy()
    image = opencv_resize(image, resize_ratio)
    # Convert to grayscale for further processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Get rid of noise with Gaussian Blur filter
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    blurred = cv2.medianBlur(blurred,7)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    erosion = cv2.erode(blurred,kernel,iterations = 1)
    # Detect white regions
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    rectKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 20))
    dilated = cv2.dilate(erosion, rectKernel)

    opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, rectKernel2)
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, rectKernel2)
    blackAndWhiteImage = cv2.adaptiveThreshold(closing,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)


    edged = cv2.Canny(blackAndWhiteImage, 30, 30, apertureSize=3)
    # Detect all contours in Canny-edged image
    contours, hierarchy = cv2.findContours(blackAndWhiteImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0,255,0), 3)
    # Get 10 largest contours
    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    image_with_largest_contours = cv2.drawContours(image.copy(), largest_contours, -1, (0,255,0), 3)
    receipt_contour = get_receipt_contour(largest_contours)
    if receipt_contour is None:
        print("Creating 4 corners from conotour failed, Returning normal image")
        s = image.shape
        receipt_contour = np.array([[[0,0]], [[0, s[0]]], [[s[1], s[0]]], [[s[1], 0]]])
    image_with_receipt_contour = cv2.drawContours(image.copy(), [receipt_contour], -1, (0, 255, 0), 2)
    
    # create birds eye view image
    scanned = wrap_perspective(original.copy(), contour_to_rect(receipt_contour, resize_ratio))        
    result = bw_scanner(scanned)
    return result, scanned

def preprocessing_otsu(file_name):
    # Sample file out of the dataset
    img = Image.open(file_name)
    img.thumbnail((800,800), Image.ANTIALIAS)
    
    image = cv2.imread(file_name)
    # Downscale image as finding receipt contour is more efficient on a small image
    resize_ratio = 1000 / image.shape[0]
    original = image.copy()
    image = opencv_resize(image, resize_ratio)
    # Convert to grayscale for further processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Get rid of noise with Gaussian Blur filter
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    blurred = cv2.medianBlur(blurred,7)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    erosion = cv2.erode(blurred,kernel,iterations = 1)
    # Detect white regions
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    rectKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 20))
    dilated = cv2.dilate(erosion, rectKernel)

    opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, rectKernel2)
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, rectKernel2)
    (thresh, blackAndWhiteImage) = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    edged = cv2.Canny(blackAndWhiteImage, 30, 30, apertureSize=3)
    # Detect all contours in Canny-edged image
    contours, hierarchy = cv2.findContours(blackAndWhiteImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0,255,0), 3)
    # Get 10 largest contours
    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    image_with_largest_contours = cv2.drawContours(image.copy(), largest_contours, -1, (0,255,0), 3)
    receipt_contour = get_receipt_contour(largest_contours)
    if receipt_contour is None:
        print("Creating 4 corners from conotour failed, Returning normal image")
        s = image.shape
        receipt_contour = np.array([[[0,0]], [[0, s[0]]], [[s[1], s[0]]], [[s[1], 0]]])
    image_with_receipt_contour = cv2.drawContours(image.copy(), [receipt_contour], -1, (0, 255, 0), 2)
    
    # create birds eye view image
    scanned = wrap_perspective(original.copy(), contour_to_rect(receipt_contour, resize_ratio))        
    result = bw_scanner(scanned)
    return result, scanned


def get_total_value(clean_total_word, ocr_words_bw_prob, ocr_bounding_box):
    try:
        for idx, word_bw_prob in enumerate(ocr_words_bw_prob[clean_total_word[0]+1:]):
            if abs(ocr_bounding_box[clean_total_word[0]+1+idx][0] - ocr_bounding_box[clean_total_word[0]][0]) < 70:
                print("something found near 50 pixels")
                continue
            else:
                next_index = ocr_words_bw_prob[clean_total_word[0]+1+idx]
                post_process_num = next_index[1].replace(',', '.').replace('S', '').replace('s', '') \
                                    .replace('$', '').replace('-', '.').replace(' ','').replace("'", '')
                
                for index, c in enumerate(reversed(post_process_num)):
                    if not c.isalnum():
                        index_true = len(post_process_num) - idx - 1
                        post_process_num = post_process_num[:index_true] + ''
                    else:
                        break
                next_index_num = float(post_process_num)
                if next_index_num > 500:
                    first_digit = int(str(next_index_num)[:1])
                    if first_digit == 8 or first_digit == 6 or first_digit == 5 or first_digit == 9 or first_digit == 7:
                        next_index_num = float(str(next_index_num)[1:])
                print(next_index_num)
                break
        if next_index[2] > 0.30:
            return next_index_num, True
        else:
            print('get_total_value not sucessfull because probability of total value is lower than 80%')
            return next_index_num, False
    except:
        print('get_total_value not sucessfull, parsing float failed!')
        return (None, False)
    
def process_sub_tax(sub_total_val, tax_val):
    if tax_val < sub_total_val:
        return sub_total_val + tax_val
    else:
        first_digit_tax = int(str(tax_val)[:1])
        if first_digit_tax == 8 or first_digit_tax == 6 or first_digit_tax == 5 or first_digit_tax == 9 or first_digit_tax == 3:
            tax_val = float(str(tax_val)[1:])
            return tax_val + sub_total_val
        else:
            return None

        
def check_if_next_value_double_digit_number(clean_total_word, ocr_words_bw):
    try:
        word_bw = ocr_words_bw[clean_total_word[0]+2]
        if word_bw.isnumeric():
            return word_bw, True
        else:
            return None, False
    except:
        return None, False

    

    
reader = easyocr.Reader(['en'])
postprocessing_words = ['Total', 'Tota', 'Tax', 'Subtotal', 'Sub', 'Sue ', 'Ttl', 'Amt Due', 'Amt ', 'Balance Due', 
                        'Balacne', 'Due', 'Tota]', 'Thru', 'totai', 'totl', 'Tdtal', 'Tothle', 'Totale', 'Totul', 
                        'Totol', 'Tatal', 'Order Tatal', 'originalttl', 'checktota', 'totel', 'TL', 'Yotal', 'payment',
                        'to-go', 'togo', 'grandtota', 'grandtotal', 'Fota', 'fotal', 'tutal']

postprocessing_words_ttl = ['total', 'tota', 'tota]','totl', 'totai', 'tdtal', 'tothle', 'totale', 'totul',
                            'totol', 'grandtotal', 'grandtota', 'tatal','Order Tatal', 'ordertatal', 'originalttl', 'checktota', 'totel',
                            'tl', 'Yotal', 'payment', 'to-go', 'togo', 'fota', 'fotal', 'tutal']
subtotal_postprocess_list = ['subtotal', 'suetotal', 'subtolal', 'subtotel']
column_names = ['file_name', 'predicted_total']
df = pd.DataFrame(columns = column_names)
for i in range(1000, 1200):
    #pd_dict = {'file_name': }
    file_name = 'C:/Users/krish/Downloads/dataset/' + str(i) + '-receipt.jpg'
    pd_dict = {'file_name': file_name, 'predicted_total': -1}
    print(file_name)
    result, scanned = preprocessing_adaptive(file_name)

    if result is None:
        result_shape = [0, 0]
    else:
        result_shape = result.shape

    if result_shape[0] < 200 or result_shape[1] < 200:
        result, scanned = preprocessing_otsu(file_name)

        result_shape = result.shape
        if result_shape[0] < 200 or result_shape[1] < 200:
            print("still less than 200 pixels")
    
#     save_name = file_name.split('/')[-1]
#     save_gray(result, save_name)
    plot_gray(result)
    ocr_result_bw = reader.readtext(result)
    ocr_bounding_box = [x[0][0] for x in ocr_result_bw]
    ocr_words_bw = [x[1] for x in ocr_result_bw]
    ocr_words_prob = [x[2] for x in ocr_result_bw]
    ocr_total_words_bw = []
    ocr_words_bw_prob = []
    for idx, x in enumerate(ocr_words_bw):
        ocr_words_bw_prob.append((idx,x, ocr_words_prob[idx]))
    print(ocr_words_bw_prob)
    
    for x in postprocessing_words:
        for idx, y in enumerate(ocr_words_bw):
            if x.lower() in y.lower():
                ocr_total_words_bw.append((idx, y, ocr_words_prob[idx]))
    
    ocr_clean_total_words_bw = []
    for total_word in ocr_total_words_bw:
        clean_word = ''.join(e for e in total_word[1] if e.isalpha()).lower()
        ocr_clean_total_words_bw.append((total_word[0], clean_word))
    ocr_clean_total_words_bw = list(set(ocr_clean_total_words_bw))
    ocr_clean_total_words_bw = sorted(ocr_clean_total_words_bw, key=lambda tup: tup[0])
    print("Found clean total words:")
    print(ocr_clean_total_words_bw)
    
    sub_total_tax_flag = False
    for index, clean_total_word in enumerate(ocr_clean_total_words_bw):
        if (('total' in clean_total_word[1]) and (clean_total_word[1] not in subtotal_postprocess_list)) or \
        ((clean_total_word[1] in 'total') or ('due' in clean_total_word[1]) or ('thru' in clean_total_word[1])) or \
        (clean_total_word[1] in postprocessing_words_ttl) and (clean_total_word[1] != 'changedue'):
            print(clean_total_word[1])
            total_val, flag = get_total_value(clean_total_word, ocr_words_bw_prob, ocr_bounding_box)            
            if ((index + 2) < len(ocr_words_bw)) and total_val is not None:
                print("Calling double digit")
                dec_double, dec_double_flag = check_if_next_value_double_digit_number(clean_total_word, 
                                                                                      ocr_words_bw)
                if dec_double_flag and dec_double is not None and total_val is not None:
                    if total_val - int(total_val) == 0:
                        total_val = float(str(int(total_val)) + '.' + str(dec_double))
            if total_val is not None and flag:
                print('Found total value: ' + str(total_val))
                pd_dict['predicted_total'] = total_val
                sub_total_tax_flag = False
                if clean_total_word[1] == 'total':
                    total_found_flag = False
                    if index + 1 == len(ocr_clean_total_words_bw):
                        break
                    for c_t in ocr_clean_total_words_bw[index+1:]:
                        if c_t[1] == 'total':
                            total_found_flag = True
                            break
                    if total_found_flag:
                        continue
                    else:
                        break
                else:
                    continue
            elif total_val is not None and not flag:
                print('Probability of output is low, will try finding tax and subtotal')
                sub_total_tax_flag = True
            else:
                print('Could not parse total value, will try finding tax and subtotal')
                sub_total_tax_flag = True
    
    if not sub_total_tax_flag:
        df = df.append(pd_dict, ignore_index=True)
        print(pd_dict)
        continue
    
    found_sub_total_flag = False
    return_org_total = False
    for clean_total_word in ocr_clean_total_words_bw:
        if (clean_total_word[1] in subtotal_postprocess_list) and sub_total_tax_flag:
            print(clean_total_word)
            sub_total_val, sub_flag = get_total_value(clean_total_word, ocr_words_bw_prob, ocr_bounding_box)
            if sub_total_val is not None and sub_flag:
                print('Found Sub total value: ' + str(sub_total_val))
                found_sub_total_flag = True
                break
            elif sub_total_val is not None and not sub_flag:
                print('Probability of sub total output is low, retruning total_value if exists')
                if total_val:
                    print('Found total value: ' + str(total_val))
                    pd_dict['predicted_total'] = total_val
                    return_org_total = True
                    break
                else:
                    print("Could not find total value")
                    pd_dict['predicted_total'] = -1
                    break
            else:
                print('Could not parse subtotal value, retruning total_value if exists')
                if total_val:
                    print('Found total value: ' + str(total_val))
                    pd_dict['predicted_total'] = total_val
                    return_org_total = True
                else:
                    print("Could not find total value")
                    pd_dict['predicted_total'] = -1
                    break
    
    if not found_sub_total_flag and total_val is not None:
        print("Could not find subtotal")
        print("Total value with low confidence: " + str(total_val))
        pd_dict['predicted_total'] = total_val
        df = df.append(pd_dict, ignore_index=True)
        print(pd_dict)
        continue
    
    if not found_sub_total_flag and return_org_total:
        print("Subtotal parsing failed or low confidence for subtotal as well")
        print("Total value with low confidence: " + str(total_val))
        pd_dict['predicted_total'] = total_val
        df = df.append(pd_dict, ignore_index=True)
        print(pd_dict)
        continue
    if not found_sub_total_flag and not return_org_total:
        print("Could not find subtotal")
        print("Could not find total")
        pd_dict['predicted_total'] = -1
        df = df.append(pd_dict, ignore_index=True)
        print(pd_dict)
        continue
    
    found_tax_flag = False
    for clean_total_word in ocr_clean_total_words_bw:
        if 'tax' in clean_total_word[1] and found_sub_total_flag and sub_total_tax_flag:
            tax_val, tax_flag = get_total_value(clean_total_word, ocr_words_bw_prob, ocr_bounding_box)
            if tax_val is not None and tax_flag:
                print('Found tax value: ' + str(tax_val))
                found_tax_flag = True
                break
            elif tax_val is not None and not tax_flag:
                print('Probability of tax output is low, retruning total_value if exists')
                if total_val:
                    print('Found total value: ' + str(total_val))
                    pd_dict['predicted_total'] = total_val
                else:
                    print("Could not find total value")
                    pd_dict['predicted_total'] = -1
            else:
                print('Could not parse tax value, retruning total_value if exists')
                if total_val:
                    print('Found total value: ' + str(total_val))
                    pd_dict['predicted_total'] = total_val
                else:
                    print("Could not find total value")
                    pd_dict['predicted_total'] = -1
    if found_tax_flag and found_sub_total_flag:
        total_value_final = process_sub_tax(sub_total_val, tax_val)
        if total_value_final is not None:
            print("Found total value: subtotal + tax: " + str(total_value_final))
            pd_dict['predicted_total'] = total_value_final
        else:
            print("can not calculate total value using sub total and tax")
            pd_dict['predicted_total'] = -1
    if not found_tax_flag:
        print("Could not find tax")
        print("Could not find total value")
        pd_dict['predicted_total'] = -1
    print(pd_dict)
    df = df.append(pd_dict, ignore_index=True)

df.to_csv('C:/Users/krish/Desktop/CeadarAssignmentCode/ceadar-data-scientist/output_csv.csv')
df_for_error_metrics = pd.read_csv('C:/Users/krish/Desktop/CeadarAssignmentCode/ceadar-data-scientist/error_metrics_csv.csv')

def calc_accuracy(df):
    count = 0
    for index, row in df.iterrows():
        if row['predicted_total'] == row['actual_total']:
            count = count + 1
    acc = (count)/(df.shape[0])
    return acc

def calc_accuracy_without_decimals(df):
    count = 0
    for index, row in df.iterrows():
        if int(row['predicted_total']) == int(row['actual_total']):
            count = count + 1
    acc = (count)/(df.shape[0])
    return acc

def number_of_images_predicted(df):
    df_new = df[df['predicted_total'] != -1]
    return df_new.shape[0]/df.shape[0]

def number_of_images_predicted_correctly(df):
    df_new = df[df['predicted_total'] != -1]
    count = 0
    for index, row in df_new.iterrows():
        if row['predicted_total'] == row['actual_total']:
            count = count + 1
    acc = (count)/(df_new.shape[0])
    return acc

def number_of_images_predicted_correctly_without_decimals(df):
    df_new = df[df['predicted_total'] != -1]
    count = 0
    for index, row in df_new.iterrows():
        if int(row['predicted_total']) == int(row['actual_total']):
            count = count + 1
    acc = (count)/(df_new.shape[0])
    return acc

print("Accuracy (Number of reciepts correctly parsed / Number of total reciepts) = " + str(calc_accuracy(df_for_error_metrics)))
print("Accuracy without decimal points (Number of reciepts correctly parsed without decimal points / Number of total reciepts) = " + str(calc_accuracy_without_decimals(df_for_error_metrics)))
print("Ratio of estimated images (Number of images which have estimations / Total number of images) = " + str(number_of_images_predicted(df_for_error_metrics)))
print("Accuracy of images which were estimated (Number of images which have correct estimations / Number of images which have estimations) = " + str(number_of_images_predicted_correctly(df_for_error_metrics)))
print("Accuracy of images which were estimated correctly without decimal points (Number of images which have correct estimations without decimal points / Number of images which have estimations) = " + str(number_of_images_predicted_correctly_without_decimals(df_for_error_metrics)))