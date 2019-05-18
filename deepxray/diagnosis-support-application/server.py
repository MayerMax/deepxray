import random
import string
import bottle
from bottle import route, request, error, template, static_file
import os
import sys

from deepxray.transferlearning.dnn import DenseLearner

root_folders = ['..', '../..', './']
root_folders = [os.path.abspath(os.path.join(x)) for x in root_folders]
for path in root_folders:
    if path not in sys.path:
        sys.path.append(path)


app = bottle.app()
model = DenseLearner.load(sys.argv[1])


@route('/')
def index():
    return template('frontend/index.html')


@route('/prediction', method='POST')
def make_preds():
    upload = request.files.get('upload')
    temp_image_path = 'frontend/{}.bmp'.format(''.join(random.choice(string.ascii_lowercase) for _ in range(5)))
    path_to_explanation = 'frontend/expl_{}.bmp'.format(
        ''.join(random.choice(string.ascii_lowercase) for _ in range(5)))
    upload.save(temp_image_path)
    predicted_class, _ = model.explain_prediction(temp_image_path, filename=path_to_explanation)
    if predicted_class == 'Норма':
        return '{} {}'.format('Normal', temp_image_path.replace('frontend/', ''))
    else:
        return '{} {}'.format('Pathology', path_to_explanation.replace('frontend/', ''))


@route('/<filename:path>')
def send_file(filename):
    return static_file(filename, root='frontend/')


def main():
    bottle.debug(True)
    bottle.run(app=app, host='localhost', port=8080)


if __name__ == "__main__":
    main()
