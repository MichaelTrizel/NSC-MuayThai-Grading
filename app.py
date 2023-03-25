import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, session
from jinja2 import Environment
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload_video'
app.config['TUTORIAL_FOLDER'] = 'tutorial'
app.secret_key = '198237645'

env = Environment()
env.globals.update(enumerate=enumerate)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/pier-selection', methods=['GET', 'POST'])
def pier_selection():

	upload_folder = './upload_video/'
	if not os.path.exists(upload_folder):
		os.makedirs(upload_folder)

	if request.method == 'POST':
		# Check if pier is selected
		if 'pier' not in request.form:
			return "Error: no pier was selected"

		# Check if file is uploaded
		if 'video' not in request.files:
			return 'No video uploaded', 400

		video_file = request.files['video']
		# Check if file is not empty
		if video_file.filename == '':
			return 'No selected file', 400

		# Save file to upload folder
		video_path = os.path.join(upload_folder, video_file.filename)
		video_file.save(video_path)

		# Render template for selected pier
		selected_pier = request.form['pier']

		# Assign Pier
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

		session['video_file'] = video_file.filename
		session['pier'] = pier

		return render_template(selected_pier + '.html', video_file=video_file.filename, pier=pier)
	else:
		return render_template('index.html')

@app.route('/upload_video/<filename>')
def video(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/tutorial/<filename>')
def tutorial(filename):
	return send_from_directory(app.config['TUTORIAL_FOLDER'], filename)

# @app.route('/upload_image/<filename>')
# def image(filename):
# 	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/loading')
def loading():
	video_file = session.get('video_file',None)
	pier = session.get('pier',None)
	if video_file is None or pier is None:
		return redirect('/')
	return render_template('loading.html', video_file=video_file, pier=pier)

@app.route('/result_1')
def result_1():
	video_file = session.get('video_file',None)
	pier = session.get('pier',None)
	if video_file is None or pier is None:
		return redirect('/')
	return render_template('result_1.html', video_file=video_file, pier=pier)

@app.route('/result_2')
def result_2():
	video_file = session.get('video_file',None)
	pier = session.get('pier',None)
	if video_file is None or pier is None:
		return redirect('/')
	return render_template('result_2.html', video_file=video_file, pier=pier)

if __name__ == '__main__':
    app.run(debug=True)
