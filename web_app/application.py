from flask import Flask, render_template, flash, request, jsonify, Markup
import os
import torch
from pathlib2 import Path
import logging
from modules.Beam_search import *
from PIL import Image
import json
from torchvision import transforms
#import settings

#set devise
device = 'cpu'
#defaults.device = torch.device('cpu')

# Load model architecture and parameters
path = Path() 

checkpoint_path = "models/Best_model.pth"

# Vocab dict loading  
vocab = json.load((path/"static/Vocab_5_cap_per_img_2_min_word_freq.json").open('rb'))
ind_str = dict(zip(vocab.values(),vocab.keys()))

# tranformation 
transformations = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5238, 0.5003, 0.4718], [0.3159, 0.3091, 0.3216])])


# set flask params
app = Flask(__name__)


app.config["IMAGE_UPLOADS"] = 'tmp'

@app.errorhandler(500)
def server_error(e):
    logging.exception('some eror')
    return """
    And internal error <pre>{}</pre>
    """.format(e), 500

@app.route("/", methods=['GET'])
def startup():
    return render_template('index.html') # pred_class

@app.route('/predict', methods=["GET",'POST'])
def predict():
    if request.method == "POST":
    #    if not request.files:
        image = request.files['file']
        upload_path = os.path.join(app.config["IMAGE_UPLOADS"], image.filename)
        image.save(upload_path)
        caps = beam_search(checkpoint_path,img_path = upload_path, beam_size = 5, vocab = vocab, transforms = transformations,device=device)
        caps = [ind_str[x] for x in caps]
        return ' '.join(caps)#jsonify(predict=str(pred_class))
                #app.logger.info("Image %s classified as %s" % (url, pred_class))
    return None



if __name__ == '__main__':
    app.run(debug=True)