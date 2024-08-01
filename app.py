from flask import Flask, render_template, request

from text_summarizer import summarizer

import topic

app=Flask(__name__)



@app.route('/')

def index():

	return render_template('index.html')



@app.route('/analyze_summarize', methods=['GET', 'POST'])

def analyze_summarize():

	if request.method == 'POST':

		rawtext = request.form['rawtext']

		summary, original_txt, len_orig_txt, len_summary=summarizer(rawtext)

	

	return render_template('summary.html', summary=summary, original_txt=original_txt, len_orig_text=len_orig_txt, len_summary = len_summary)

@app.route('/analyze_topic', methods=['GET', 'POST'])
def analyze_topic():
   topic_label = None
   if request.method == 'POST':
     input_text = request.form['rawtext']
     topic_label = topic.classify_text(input_text)
     topic_label = topic_label.upper()

   return render_template('topic.html', input_text=input_text,topic_label=topic_label)
    

if __name__ == "__main__":

	app.run(debug=True)