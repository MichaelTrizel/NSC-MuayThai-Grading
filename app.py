import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, session
from jinja2 import Environment

import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from PIL import Image

from Converter import Conveter
from MuayThai import MuayThai

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload_video'
app.config['TUTORIAL_FOLDER'] = 'tutorial'
app.config['IMAGE_FOLDER'] = 'image_assets'
app.config['POSE_IMAGES'] = 'pose_images'
app.secret_key = '198237645'

env = Environment()
env.globals.update(enumerate=enumerate)

def define_pier(selected_pier):
	if selected_pier == "pier1":
		pier = "ท่าสลับฟันปลา"
	elif selected_pier == "pier2":
		pier = "ท่าตาเถรค้ำพัก"
	elif selected_pier == "pier3":
		pier = "ท่ามอญยันหลัก"
	elif selected_pier == "pier4":
		pier = "ท่าดับชวาลา"
	elif selected_pier == "pier5":
		pier = "ท่าหักคอเอราวัณ"
	return pier

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/validate', methods=['GET', 'POST'])
def validate():
	upload_folder = './upload_video/'
	if not os.path.exists(upload_folder):
		os.makedirs(upload_folder)

	if request.method == 'POST':
		# Check if pier is selected
		if 'pier' not in request.form:
			error_message = 'กรุณาเลือกท่ามวยไทยที่ต้องการประเมินด้วย'
			return render_template('error.html', error_message=error_message)
		
		# Check if file is uploaded
		if 'video' not in request.files:
			error_message = 'กรุณาอัปโหลดคลิปที่ต้องการประเมินด้วย'
			return render_template('error.html', error_message=error_message)
		
		# Check if file is not empty
		video_file = request.files['video']
		if video_file.filename == '':
			error_message = 'กรุณาอัปโหลดคลิปที่ต้องการประเมินด้วย'
			return render_template('error.html', error_message=error_message)

		# Save file to upload_video folder + rename the file
		num_files = len([f for f in os.listdir('upload_video') if os.path.isfile(os.path.join('upload_video', f))])
		video_name = str(num_files + 1) + '.MOV'
		video_path = os.path.join(upload_folder, video_name)
		video_file.save(video_path)

		# Define the selected pier
		selected_pier = request.form['pier']
		pier = define_pier(selected_pier)

		# Put the value into session
		session['selected_pier'] = selected_pier
		session['pier'] = pier
		session['video_path'] = video_path
		session['video_file'] = video_name

		return render_template('pier.html', video_file=video_name, pier=pier)
	else:
		return render_template('index.html')

@app.route('/upload_video/<filename>')
def upload_video(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/tutorial/<filename>')
def tutorial(filename):
	return send_from_directory(app.config['TUTORIAL_FOLDER'], filename)

@app.route('/image/<filename>')
def image(filename):
	return send_from_directory(app.config['IMAGE_FOLDER'], filename)

@app.route('/pose_images/<filename>')
def pose_images(filename):
	return send_from_directory(app.config['POSE_IMAGES'], filename)

def find_missing(lst, step):
	step_missed = []
	for i in range(1, step+1):
		if lst.count(i) == 0:
			step_missed.append(i)
	return step_missed

@app.route('/loading')
def loading():
	return render_template('loading.html')

@app.route('/grading')
def grading():
	return render_template('grading.html')

@app.route('/check')
def check():
	video_path = session.get('video_path',None)

	# Model code
	user_folder = './user_data/'
	if not os.path.exists(user_folder):
		os.makedirs(user_folder)

	user_converter = Conveter(video_path, user_folder)
	clip_df, pose_images = user_converter.convert_video_to_node()
	user_converter.write_video_node_to_csv(clip_df, 'user.csv')

	# Create folder to store pose_images image
	pose_images_folder = './pose_images'
	if not os.path.exists(pose_images_folder):
		os.makedirs(pose_images_folder)

	# Save each ndarray as an image file
	image_filenames = []
	for i, image in enumerate(pose_images):
		# Convert the color channels to RGB order by reversing the order of the last axis
		image = image[..., ::-1]

		# Save ndarray to image and save to folder
		image_filename = f'{pose_images_folder}/pose{i}.jpg'
		Image.fromarray(image).save(image_filename)
		image_filenames.append('pose' + str(i) + '.jpg')
	
	return render_template('check.html', pose_images=image_filenames)

@app.route('/result')
def result():
	video_file = session.get('video_file',None)
	pier = session.get('pier',None)
	video_path = session.get('video_path',None)

	if pier == "ท่าสลับฟันปลา":
		true_df = pd.read_csv('./trainer/1/true.csv')
		cal_df = pd.read_csv('./trainer/1/cal.csv')
		trainer_df = pd.read_csv('./trainer/1/clip.csv')
		true_backup_df = pd.read_csv('./trainer/1/true_backup.csv')

		#Dtw
		number_pier = 1
		distance_point = 50

		#Grading
		grade_df = pd.read_csv('./trainer/1/grade.csv')
		mean = 41.3333303030303
		std = 32.76412259432642

	elif pier == "ท่าตาเถรค้ำพัก":
		true_df = pd.read_csv('./trainer/6/true.csv')
		cal_df =pd.read_csv('./trainer/6/cal.csv')
		trainer_df = pd.read_csv('./trainer/6/clip.csv')
		true_backup_df = pd.read_csv('./trainer/6/true_backup.csv')
		failed_df = pd.read_csv('./trainer/6/failed.csv')

		#Dtw
		number_pier = 6
		distance_point = 160

		#Grading
		grade_df = pd.read_csv('./trainer/6/grade.csv')
		mean = 41.3333303030303
		std = 32.76412259432642

	elif pier == "ท่ามอญยันหลัก":
		true_df = pd.read_csv('./trainer/7/true.csv')
		cal_df = pd.read_csv('./trainer/7/cal.csv')
		trainer_df = pd.read_csv('./trainer/7/clip.csv')
		true_backup_df = pd.read_csv('./trainer/7/true_backup.csv')
		failed_df = pd.read_csv('./trainer/7/failed.csv')

		#Dtw
		number_pier = 7
		distance_point = 50

		#Grading
		grade_df = pd.read_csv('./trainer/7/grade.csv')
		mean = 41.3333303030303
		std = 32.76412259432642

	elif pier == "ท่าดับชวาลา":
		true_df = pd.read_csv('./trainer/13/true.csv')
		cal_df = pd.read_csv('./trainer/13/cal.csv')
		trainer_df = pd.read_csv('./trainer/13/clip.csv')
		true_backup_df = pd.read_csv('./trainer/13/true_backup.csv')
		failed_df = pd.read_csv('./trainer/13/failed.csv')

		#Dtw
		number_pier = 13
		distance_point = 150

		#Grading
		grade_df = pd.read_csv('./trainer/13/grade.csv')
		mean = 41.3333303030303
		std = 32.76412259432642

	elif pier == "ท่าหักคอเอราวัณ":
		true_df = pd.read_csv('./trainer/15/true.csv')
		cal_df = pd.read_csv('./trainer/15/cal.csv')
		trainer_df = pd.read_csv('./trainer/15/clip.csv')
		true_backup_df = pd.read_csv('./trainer/15/true_backup.csv')
		failed_df = pd.read_csv('./trainer/15/failed.csv')

		#Dtw
		number_pier = 15
		distance_point = 150

		#Grading
		grade_df = pd.read_csv('./trainer/15/grade.csv')
		mean = 41.3333303030303
		std = 32.76412259432642

	del true_df['Unnamed: 0']
	del cal_df['Unnamed: 0']
	del failed_df['Unnamed: 0']

	user_df = pd.read_csv('./user_data/user.csv')
	del user_df['Unnamed: 0']
	user_muay = MuayThai(video_path, user_df, 4, true_df, cal_df)
	user_point, user_true_frames, user_true_angles, user_true_steps, user_failed_steps = user_muay.check()
	#print(user_point)

	# Seperate to each result page
	# This is result_2 
	if (user_point < user_muay.step):
		print('bad')

		user_backup_muay = MuayThai(video_path, user_df, 4, true_backup_df, cal_df)
		user_backup_point, user_backup_true_frames, user_backup_true_angles, user_backup_true_steps, user_backup_failed_steps = user_backup_muay.check()

		if (user_backup_point < user_backup_muay.step):
			step_missed = find_missing(user_backup_true_steps, 4)
			
			#Answer
			step_missed_str = ', '.join(map(str, step_missed))
			print(step_missed_str)
			print(user_backup_failed_steps)

			# Get detail for failed step
			failed_info = {}
			for step, ls_sub in user_backup_failed_steps.items():
				if len(ls_sub) != 0:
					temp = []
					for sub_step in ls_sub:
						failed_label = failed_df.loc[(failed_df['step'] == step) & (failed_df['sub_step'] == sub_step), 'failed_label'].iloc[0]
						temp.append(failed_label)
					failed_info[step] = temp
			# print(failed_info)

			return render_template('result_2.html', video_file=video_file, pier=pier, step_missed_str=step_missed_str, failed_info=failed_info)
		
		user_true_angles = user_backup_true_angles
	
	# This is result_1
	print('good')
	max_angles_len_user = len(max(user_true_angles, key=len))
	user_padding_angle = [np.pad(arr, 
                         pad_width=max_angles_len_user-len(arr), 
                         mode='constant', 
                         constant_values=0)[max_angles_len_user-len(arr):] for arr in user_true_angles]
	clip_max = 35
	point_criterion = 0
	for i in range(1, clip_max+1):
		trainer_muay = MuayThai(''.format(i), 
						trainer_df[trainer_df['clip_name'] == '{}_{}'.format(number_pier, i)], 
						4, true_df, cal_df)
		
		trainer_point, trainer_true_frames, trainer_true_angles, trainer_true_steps, trainer_failed_steps = trainer_muay.check()
		max_angles_len_train = len(max(trainer_true_angles, key=len))
		trainer_padding_angle = [np.pad(arr, 
								pad_width=max_angles_len_train-len(arr), 
								mode='constant', 
								constant_values=0)[max_angles_len_train-len(arr):] for arr in trainer_true_angles]
		
		distance, path = fastdtw(trainer_padding_angle, user_padding_angle, dist=euclidean)
		if (distance <= distance_point):
			point_criterion += 1

	#Answer of similarity
	similarity = (point_criterion/clip_max)*100
	print(similarity)

	#Answer of grade
	z_score = (similarity-mean)/std
	t_score = (z_score*10)+50
	print('T_score', t_score)

	grade = 'N'
	if t_score >= grade_df.loc[grade_df['grade'] == 'A', 'min'].iloc[0]:
		grade = 'A'
	elif t_score >= grade_df.loc[grade_df['grade'] == 'B', 'min'].iloc[0] and t_score < grade_df.loc[grade_df['grade'] == 'B', 'max'].iloc[0]:
		grade = 'B'
	elif t_score >= grade_df.loc[grade_df['grade'] == 'C', 'min'].iloc[0] and t_score < grade_df.loc[grade_df['grade'] == 'C', 'max'].iloc[0]:
		grade = 'C'
	elif t_score < grade_df.loc[grade_df['grade'] == 'D', 'max'].iloc[0]:
		grade = 'D'
	print(grade)

	similarity = round(similarity, 2)
	return render_template('result_1.html', video_file=video_file, pier=pier, similarity=similarity, grade=grade)
	#End of model code

if __name__ == '__main__':
    app.run(debug=True)
