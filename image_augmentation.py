import numpy as np
import cv2
import argparse
import random
import os
import glob
import csv
import datetime

from PIL import Image

import xml.etree.ElementTree as ET


def draw_box(image, box):

	# box : x1, x2, y1, y2
	image = cv2.rectangle(image, (int(box[0]), int(box[2])), (int(box[1]), int(box[3])), (255,0,0), 3)
	
	return image


def noise(image, random_num):

	noise_image = image.copy()

	# scale = 0.05
	if random_num == 1:
		noised_image = np.clip((noise_image/255 + np.random.normal(scale=0.05, size=noise_image.shape)) * 255, 0, 255).astype('uint8')

	# scale = 0.1
	if random_num == 2:
		noised_image = np.clip((noise_image/255 + np.random.normal(scale=0.1, size=noise_image.shape)) * 255, 0, 255).astype('uint8')

	# salt and pepper
	if random_num == 3:
		noise_image.astype(np.float16, copy = False)
		noise_image = np.multiply(noise_image, (1 / 255))
		salt_and_pepper = 300
		noise = np.random.randint(salt_and_pepper, size = (noise_image.shape[0], noise_image.shape[1], 1))
		noise_image = np.where(noise == 0, 0, noise_image)
		noise_image = np.where(noise == (salt_and_pepper - 1), 1, noise_image)
		noised_image = cv2.convertScaleAbs(noise_image, alpha = (255 / 1))

	return noised_image


def blur(image, random_num):

	blur_image = image.copy()

	# filter = 5 x 5
	if random_num == 1:
		blured_image = cv2.blur(blur_image, (5,5))

	# filter = (3 x 3) x 2
	if random_num == 2:
		blured_image = cv2.blur(blur_image, (3,3))
		blured_image = cv2.blur(blur_image, (3,3))

	# filter = (3 x 3) x 3
	if random_num == 3:
		blured_image = cv2.blur(blur_image, (3,3))
		blured_image = cv2.blur(blur_image, (3,3))
		blured_image = cv2.blur(blur_image, (3,3))

	return blured_image


def rgb_to_hsv(rgb):

	rgb = rgb.astype('float')
	hsv = np.zeros_like(rgb)

	hsv[..., 3:] = rgb[..., 3:]
	r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
	maxc = np.max(rgb[..., :3], axis=-1)
	minc = np.min(rgb[..., :3], axis=-1)
	hsv[..., 2] = maxc
	mask = maxc != minc
	hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
	rc = np.zeros_like(r)
	gc = np.zeros_like(g)
	bc = np.zeros_like(b)
	rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
	gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
	bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
	hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
	hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0

	return hsv


def hsv_to_rgb(hsv):

	rgb = np.empty_like(hsv)
	rgb[..., 3:] = hsv[..., 3:]
	h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
	i = (h * 6.0).astype('uint8')
	f = (h * 6.0) - i
	p = v * (1.0 - s)
	q = v * (1.0 - s * f)
	t = v * (1.0 - s * (1.0 - f))
	i = i % 6
	conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
	rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
	rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
	rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)

	return rgb.astype('uint8')


def hue(image, random_num):

	hue_image = image.copy()

	if random_num == 1:
		for amount in (1, 50):
			amount = amount/360.
			arr = np.array(hue_image)
			hsv = rgb_to_hsv(arr)
			hsv[..., 0] = (hsv[..., 0]+amount) % 1.0
			rgb_hue = hsv_to_rgb(hsv)

	if random_num == 2:
		for amount in (1, 70):
			amount = amount/360.
			arr = np.array(hue_image)
			hsv = rgb_to_hsv(arr)
			hsv[..., 0] = (hsv[..., 0]+amount) % 1.0
			rgb_hue = hsv_to_rgb(hsv)

	if random_num == 3:
		for amount in (50, 133):
			amount = amount/360.
			arr = np.array(hue_image)
			hsv = rgb_to_hsv(arr)
			hsv[..., 0] = (hsv[..., 0]+amount) % 1.0
			rgb_hue = hsv_to_rgb(hsv)

	return rgb_hue


def convolution(image, random_num):

	convolution_image = image.copy()

	if random_num == 1:
		kernel = np.ones((3,3), np.float32) / (3 * 2)
		convolution_image = cv2.filter2D(convolution_image, -1, kernel)

	if random_num == 2:
		kernel = np.ones((3,3), np.float32) / (3 * 4)
		convolution_image = cv2.filter2D(convolution_image, -1, kernel)

	if random_num == 3:
		kernel = np.ones((5,5), np.float32) / (3 * 5)
		convolution_image = cv2.filter2D(convolution_image, -1, kernel)

	return convolution_image


def flip(image):

	flip_image = image.copy()
	flip_image_rgb = cv2.cvtColor(flip_image, cv2.COLOR_BGR2RGB)
	fliped_image = cv2.flip(flip_image, 1)

	return fliped_image


def flip_box_coordinate(b, w, h):

	bb = []

	flip_x1 = int(w - b[0])
	flip_x2 = int(w - b[1])
	flip_y1 = int(b[2])
	flip_y2 = int(b[3])

	# xmin, xmax, ymin, ymax
	bb = ((flip_x2, flip_x1, flip_y1, flip_y2))

	return bb


def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):



	if brightness != 0:
		if brightness > 0:
			shadow = brightness
			highlight = 255
		else:
			shadow = 0
			highlight = 255 + brightness
		alpha_b = (highlight - shadow)/255
		gamma_b = shadow

		buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
	else:
		buf = input_img.copy()

	if contrast != 0:
		f = 131*(contrast + 127)/(127*(131-contrast))
		alpha_c = f
		gamma_c = 127*(1-f)

		buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

	return buf




def get_file_list(input_dir):

	filenames = []

	files = glob.glob(input_dir + '*.jpg')

	return files


def get_xml_path(fpath):

	xml_path = []

	xml_path = fpath
	xml_path = xml_path[:-4]
	xml_path = xml_path + '.xml'

	return xml_path

def save_jpg_xml(fpath, fxml_name, fxml_path, image, tree, flag):

	save_path = fpath
	save_path = save_path[:-4]

	suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

	# e.g. 
	# origin name : GooLying00738
	# saved  name : GooLying00738_b190607111728

	jpg_filename = save_path + "_" + flag + suffix + ".jpg"
	xml_filename = save_path + "_" + flag + suffix + ".xml"

	fxml_name = fxml_name[:-4]
	fxml_name = fxml_name + "_" + flag + suffix + ".xml"

	fxml_path = fxml_path[:-4]
	fxml_path = fxml_path + "_" + flag + suffix + ".xml"

	root = tree.getroot()
	root.find('filename').text = fxml_name
	root.find('path').text = fxml_path

	cv2.imwrite(jpg_filename, image)

	tree.write(xml_filename)

def Augmentation(input_dir):


	# Get file
	files = get_file_list(input_dir)

	for i in range(len(files)):


		classes = ["person"]
		total_box = []
		flip_box = []

		random_num = random.randrange(1, 4) 

		xml_path = get_xml_path(files[i])

		# Read jpg file
		original_image = cv2.imread(files[i])



		# Read xml file
		fxml = open(xml_path)
		tree = ET.parse(fxml)
		root = tree.getroot()

		size = root.find('size')
		w = int(size.find('width').text)
		h = int(size.find('height').text)

		fxml_name = root.find('filename').text
		fxml_path = root.find('path').text




		# Save total box
		for obj in root.iter('object'):

			cls = obj.find('name').text
			if cls not in classes : 
				continue

			xmlbox = obj.find('bndbox')

			# xmin, xmax, ymin, ymax
			b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))

			total_box.append(b)


		


		# blur
		blur_image = blur(original_image, random_num)
		save_jpg_xml(files[i], fxml_name, fxml_path, blur_image, tree, 'b')


		# noise
		noise_image = noise(original_image, random_num)
		save_jpg_xml(files[i], fxml_name, fxml_path, noise_image, tree, 'n')


		conv_image = convolution(original_image, random_num)
		save_jpg_xml(files[i], fxml_name, fxml_path, conv_image, tree, 'c')

		bright_image = apply_brightness_contrast(original_image, -127, 0)
		save_jpg_xml(files[i], fxml_name, fxml_path, bright_image, tree, 'a')


		# Flip
		flip_image = flip(original_image)	

		# Get flip box coordinate
		for a in range(len(total_box)):
			bb = flip_box_coordinate(total_box[a], w, h)
			flip_box.append(bb)

		flip_cnt = 0 


		# Updata xml file
		for obj in root.iter('object'):

			cls = obj.find('name').text

			if cls not in classes : 
				continue

			xmlbox = obj.find('bndbox')
			xmlbox.find('xmin').text = str(flip_box[flip_cnt][0])
			xmlbox.find('xmax').text = str(flip_box[flip_cnt][1])
			xmlbox.find('ymin').text = str(flip_box[flip_cnt][2])
			xmlbox.find('ymax').text = str(flip_box[flip_cnt][3])

			flip_cnt = flip_cnt + 1

		# Save jpg, xml
		save_jpg_xml(files[i], fxml_name, fxml_path, flip_image, tree, 'f')


		print("Augmentation @@@ ", random_num," ", files[i])


		


if __name__ == '__main__':


	# image folder
	input_dir = "/home/seohee/Augmentation/lying/"

	# e.g. 
	# 1. Randomly  Augmentation : python3 augment.py -f 0
	# 2. Selective Augmentation : python3 augment.py -f 1 (or 2 or 3 or 4 or 5 or 6)
	
	#parser = argparse.ArgumentParser()
	#parser.add_argument(
	#	'-f', '--filter', type=int, default='0', help='0:random, 1:blur, 2:noise, 3:hue, 4:convolution, 5:flip, 6:translation')
	#parser.add_argument('-f', '--filter', type=int,default=1, help='fliter')
	#args = parser.parse_args()

	#Augmentation(args, input_dir)
	Augmentation(input_dir)

