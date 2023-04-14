import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, session
from jinja2 import Environment

import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from Converter import Conveter
from MuayThai import MuayThai

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload_video'
app.config['TUTORIAL_FOLDER'] = 'tutorial'
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

@app.route('/video', methods=['GET', 'POST'])
def video():
	selected_pier = request.form['pier']
	session['selected_pier'] = selected_pier
	pier = define_pier(selected_pier)
	session['pier'] = pier

	video_method = request.form['video_method']
	if video_method == "method1":
		return render_template('uploaded.html', pier=pier)
	elif video_method == "method2":
		return render_template('record.html', pier=pier)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
	upload_folder = './upload_video/'
	if 'video' not in request.files:
		return 'No video file uploaded', 400
	
	video = request.files['video']
	folder_path = 'upload_video'
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

	num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
	# print(f'The folder contains {num_files} files.')
	video_file = str(num_files + 1) + '.MOV'
	video_path = os.path.join(upload_folder, video_file)
	print('Upload Section')
	print(video_path)
	print(video_file)

	session['video_path'] = video_path
	session['video_file'] = video_file

	video.save(os.path.join(folder_path, video_file))

	return 'Woking'

@app.route('/validate_record', methods=['GET', 'POST'])
def validate_record():
	selected_pier = session.get('selected_pier', None)
	pier = session.get('pier', None)

	video_file = session.get('video_file', None)

	# folder_path = 'upload_video'
	# num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
	# video_file = str(num_files) + '.MOV'

	print('val_rec')
	print(video_file)

	return render_template(selected_pier + '.html', video_file=video_file, pier=pier)

@app.route('/validate_upload', methods=['GET', 'POST'])
def validate_upload():
	upload_folder = './upload_video/'
	if not os.path.exists(upload_folder):
		os.makedirs(upload_folder)

	selected_pier = session.get('selected_pier',None)
	pier = session.get('pier',None)

	if request.method == 'POST':
		# Check if file is uploaded
		if 'video' not in request.files:
			return render_template('error2.html')

		video_file = request.files['video']
		# Check if file is not empty
		if video_file.filename == '':
			return render_template('error2.html')

		# Save file to upload folder
		num_files = len([f for f in os.listdir('upload_video') if os.path.isfile(os.path.join('upload_video', f))])
		# print(f'The folder contains {num_files} files.')
		video_name = str(num_files + 1) + '.MOV'
		video_path = os.path.join(upload_folder, video_name)
		video_file.save(video_path)
		# print(video_file.filename)
		# print(video_path)

		session['video_path'] = video_path
		session['video_file'] = video_name
		# print(video_path)

		return render_template(selected_pier + '.html', video_file=video_name, pier=pier)
	else:
		return render_template('index.html')

@app.route('/upload_video/<filename>')
def upload_video(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/tutorial/<filename>')
def tutorial(filename):
	return send_from_directory(app.config['TUTORIAL_FOLDER'], filename)

# @app.route('/upload_image/<filename>')
# def image(filename):
# 	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def find_missing(lst, step):
	step_missed = []
	for i in range(1, step+1):
		if lst.count(i) == 0:
			step_missed.append(i)
	return step_missed

@app.route('/saving')
def saving():
	return render_template('saving.html')

@app.route('/loading')
def loading():
	return render_template('loading.html')

@app.route('/result')
def result():
	video_file = session.get('video_file',None)
	pier = session.get('pier',None)
	video_path = session.get('video_path',None)

	print('Result Sec')
	print(video_file)
	print(video_path)
	print(pier)

	# if video_file is None or pier is None:
	# 	return redirect('/')
	# return render_template('loading.html', video_file=video_file, pier=pier)

	#Model code
	user_folder = './user_data/'
	if not os.path.exists(user_folder):
		os.makedirs(user_folder)

	user_converter = Conveter(video_path, user_folder)
	clip_df, pose_images = user_converter.convert_video_to_node()
	user_converter.write_video_node_to_csv(clip_df, 'user.csv')

	# Your management


	# End

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

		#Dtw
		number_pier = 15
		distance_point = 150

		#Grading
		grade_df = pd.read_csv('./trainer/15/grade.csv')
		mean = 41.3333303030303
		std = 32.76412259432642

	del true_df['Unnamed: 0']
	del cal_df['Unnamed: 0']

	user_df = pd.read_csv('./user_data/user.csv')
	del user_df['Unnamed: 0']
	user_muay = MuayThai(video_path, user_df, 4, true_df, cal_df)
	user_point, user_true_frames, user_true_angles, user_true_steps, user_failed_steps = user_muay.check()
	#print(user_point)

	# Seperate to each result page
	# This is result_2 
	if (user_point < user_muay.step):
		print('bad1')

		user_backup_muay = MuayThai(video_path, user_df, 4, true_backup_df, cal_df)
		user_backup_point, user_backup_true_frames, user_backup_true_angles, user_backup_true_steps, user_backup_failed_steps = user_backup_muay.check()

		if (user_backup_point < user_backup_muay.step):
			step_missed = find_missing(user_backup_true_steps, 4)
			
			#Answer
			step_missed_str = ', '.join(map(str, step_missed))
			print(step_missed_str)
			print(user_backup_failed_steps)

			return render_template('result_2.html', video_file=video_file, pier=pier, step_missed_str=step_missed_str)
		
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
		
		trainer_point, trainer_true_frames, trainer_true_angles, trainer_true_steps = trainer_muay.check()
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

# @app.route('/result_1')
# def result_1():
# 	video_file = session.get('video_file',None)
# 	pier = session.get('pier',None)
# 	if video_file is None or pier is None:
# 		return redirect('/')
# 	return render_template('result_1.html', video_file=video_file, pier=pier)

# @app.route('/result_2')
# def result_2():
# 	video_file = session.get('video_file',None)
# 	pier = session.get('pier',None)
# 	if video_file is None or pier is None:
# 		return redirect('/')
# 	return render_template('result_2.html', video_file=video_file, pier=pier)

if __name__ == '__main__':
    app.run(debug=True)
