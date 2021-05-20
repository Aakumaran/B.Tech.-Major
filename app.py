from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import shutil
from pathlib import Path
import uuid
import numpy
from util_2 import *
from util_4 import *
from werkzeug.urls import url_parse
import wikipedia

app = Flask(__name__) 


@app.route('/')
def topic():
	return render_template('topic.html')

@app.route('/analysis', methods =['GET', 'POST'])
def analysis():
	url = request.referrer
	url = str((url_parse(url).path)[1:])
	if url == "":
		topic_name[0] = request.form["topic"]
		max_marks[0] = request.form["max_marks"]
		if topic_name[0] == "":
			return redirect(request.referrer)
		if max_marks[0]=="":
			max_marks[0]='100'  #6.4 default
		word_count = request.form["b"]
		if word_count=="":
			word_count='100'  #6.4 default
		w_len = request.form["c"]
		if w_len=="":
			w_len='1'
		w_struct = request.form["d"]
		if w_struct=="":
			w_struct='1'
		w_gram = request.form["e"]
		if w_gram=="":
			w_gram='1'
		w_fact = request.form["f"]
		if w_fact=="":
			w_fact='1'
		

		ui[0] = int(w_len)
		ui[1] = int(w_struct)
		ui[2] = int(w_gram)
		ui[3] = int(w_fact)
		ui[4] = int(word_count)
		ui[5] = int(max_marks[0])
		script_location = os.path.dirname(os.path.abspath("lstm_model.h5"))
		test_file_dir = os.listdir(f"{script_location}\\test_files")
		test_file_dir.sort()
	script_location = os.path.dirname(os.path.abspath("lstm_model.h5"))
	test_file_dir = os.listdir(f"{script_location}\\test_files")
	test_file_dir.sort()
	return render_template('analysis.html', one_dir = test_file_dir)

@app.route('/content', methods =['GET', 'POST'])

def submit_topic():
	if request.method == "POST":
		if topic_name[0] == "":
			return redirect(url_for('topic'))
		url = request.referrer
		url = str((url_parse(url).path)[1:])
		if url == "analysis":
			try:
				analyser = request.form["check"]
				if analyser == 'factual':
					# script_location = Path(__file__).absolute().parent
					script_location = os.path.dirname(os.path.abspath("lstm_model.h5"))
					source = request.form["options"]
					if source == "Select":
						return redirect(request.referrer)
					elif source == "Wikipedia":
						ref[0] = topic_name[0] + '-wiki' + '.txt'
						flag[0] = 1
						try:
							page_object=wikipedia.page(topic_name[0])
							wiki_content=page_object.content
						except wikipedia.DisambiguationError as e:
                                                        s = random.choice(e.options)
                                                        page_object = wikipedia.page(s)
                                                        wiki_content=page_object.content
						except:
							return ('WIKIPEDIA PAGE NOT AVAILABLE FOR GIVEN TOPIC(Try Another Method)', 400)
						file_location = f"{script_location}\\test_files\\{ref[0]}"
						with open(file_location, 'w', errors='ignore') as f3:
							f3.writelines(wiki_content)
						f3.close()
					elif source == "Existing_File":
						flag[0] = 2
						try:
							e_dir = request.form["test_dir"]
							ref[0] = e_dir
						except Exception as e:
							return('',204)
					elif source == "Input_Data":
						flag[0] = 2
						u_file = request.files['input_file']
						if str(u_file)=="<FileStorage: '' ('application/octet-stream')>":
							return redirect(request.referrer)
						u_filename = secure_filename(u_file.filename)
						upload_filename = topic_name[0] + '-user' + '.txt'
						u_file.save(os.path.join("test_files",u_filename))
						u_file_location = f"{script_location}\\test_files\\{u_filename}"
						u_filelocation = f"{script_location}\\test_files\\{upload_filename}"
						with open(u_file_location, errors='ignore') as f4:
							u_file_content = f4.read()
						with open(u_filelocation, 'w', errors='ignore') as f5:
							f5.writelines(u_file_content)
						f4.close()
						f5.close()
						os.remove(u_file_location)
						ref[0] = upload_filename
				elif analyser == 'structural':
					flag[0] = 0
					ref[0] = "null"
			except:
				return redirect(request.referrer)
		
		a = data[0]
		marks = max_marks[0]
		
		if 'submit' in request.form:
			data[1] = submit_textarea()
		elif 'upload' in request.form:
			data[1] = getfile()
		else:
			data[1] = ""
		
		topic = topic_name[0]
		marks = max_marks[0]
		a = data[1]
		data[1] = ''
		s=k[0]
		k[0]=0
		
		return render_template('index.html', k = s, data = a, topic=topic, marks=marks)

def submit_textarea():
	if request.method == "POST": 
		#content typed
		text = request.form["text"]
		filename = str(uuid.uuid4())+'.txt'
		main_name[0] = filename
		main_name[1] = filename
		script_location = os.path.dirname(os.path.abspath("lstm_model.h5"))
		file_location = f"{script_location}\\files\\{filename}"
		with open(file_location, 'w', errors='ignore') as fp: 
			fp.writelines(text)
		# k is the output from the learning algorithm
		if text =="":
			k[0] = 0
		else:
			result[0]=marks(ui, flag[0], main_name[0], ref[0])
			k[0] = round(result[0][0],2)
		if numpy.isnan(k):
			k[0] = 0
		fp.close()
		return text




def getfile():
	if request.method == 'POST':
		file = request.files['myfile']
		if str(file)=="<FileStorage: '' ('application/octet-stream')>":
			text = ''
			return text
		filename = secure_filename(file.filename) 
		main_name[1] = filename
		file.save(os.path.join("files",filename))
		script_location = os.path.dirname(os.path.abspath("lstm_model.h5"))
		file_location = f"{script_location}\\files\\{filename}"
		name = str(uuid.uuid4())
		file_name = name +'.txt' 
		main_name[0] = file_name
		file.save(os.path.join("files",file_name))
		filelocation = f"{script_location}\\files\\{file_name}"
		with open(file_location, errors='ignore') as f:
			#content from uploaded file
			file_content = f.read()
		with open(filelocation, 'w', errors='ignore') as f1:
			f1.writelines(file_content)
		
		text = file_content
		f.close()
		f1.close()
		if Path(filename).suffix == '.txt':
			os.remove(file_location)
		else:
			rename = name + Path(filename).suffix
			rename_location = f"{script_location}\\files\\{rename}"
			os.rename(file_location, rename_location)
		# k is the output from the learning algorithm
		
		if file_content == None:
			k[0] = 0
		else:
			result[0]=marks(ui, flag[0], main_name[0], ref[0])
			k[0] = round(result[0][0],2)
		if numpy.isnan(k):
			k[0] = 0
		return text


@app.route('/feedback', methods=['GET','POST'])
def feedback():
	length = round(result[0][1][0][0],2)
	structure = round(result[0][1][1][0],2)
	grammer = round(result[0][1][2][0],2)
	factual = round(result[0][1][3][0],2)
	sentiment = round(result[0][1][4][0],2)
	rem_length = str(result[0][1][0][1])
	rem_structure = str(result[0][1][1][1])
	rem_grammer = str(result[0][1][2][1])
	rem_factual = str(result[0][1][3][1])
	rem_sentiment = str(result[0][1][4][1])
	return render_template('feedback.html', length=length, structure=structure, grammer=grammer, factual=factual, sentiment=sentiment, rem_length=rem_length, rem_structure=rem_structure, rem_grammer=rem_grammer, rem_factual=rem_factual, rem_sentiment=rem_sentiment)


@app.route('/compare', methods=['GET','POST'])
def compare():
	try:
		if request.method == "POST":
			files = request.files.getlist("file[]")
			if str(files[0])=="<FileStorage: '' ('application/octet-stream')>":
				return redirect(request.referrer)
			else:
				for i in range(len(files)):
					filename = secure_filename(files[i].filename)
					files[i].save(os.path.join("test_files_compare",filename))

			table = check_plagiarism(str(main_name[0]))
			table=list(table)
			table_data = list()
			for data in table:
				src_file = main_name[1]
				test_file=list(data)[1].split('\\')
				test_file = str(test_file[-1])
				result = round(float(list(data)[2]),2)*100
				table_data.extend([src_file,test_file,f'{result}%']) 

			script_location = os.path.dirname(os.path.abspath("lstm_model.h5"))
			dirpath = f"{script_location}\\test_files_compare"
			for filename in os.listdir(dirpath):
				filepath = os.path.join(dirpath, filename)
				try:
					shutil.rmtree(filepath)
				except OSError:
					os.remove(filepath)
			return render_template('result.html', table_data = table_data)
	except:
		script_location = os.path.dirname(os.path.abspath("lstm_model.h5"))
		dirpath = f"{script_location}\\test_files_compare"
		filpath = f"{script_location}\\files"
		for filename in os.listdir(dirpath):
			filepath = os.path.join(dirpath, filename)
			try:
				shutil.rmtree(filepath)
			except OSError:
				os.remove(filepath)
		for filename in os.listdir(filpath):
			filepath = os.path.join(filpath, filename)
			try:
				shutil.rmtree(filepath)
			except OSError:
				os.remove(filepath)
		return ('File Not Readable', 400)
  
if __name__=='__main__':
	main_name = ['a','b']#name of the file uploaded and name given to it by code  
	k = [0] #score
	data = ['','']#text entered
	topic_name = [''] #topic_name
	max_marks = [0] #maximum_marks
	result = ['']
	ui = [0]*6
	flag = [0]
	ref = ['']
	test_file_dir = ['']
	app.run(host="localhost")
