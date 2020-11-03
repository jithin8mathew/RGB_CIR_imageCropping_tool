#!/usr/bin/env python
# coding: utf-8

# In[3]:


import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
from dash_canvas import DashCanvas
import json
from dash_table import DataTable
from PIL import Image
import PIL
import os
from dash_canvas.utils import (array_to_data_url, parse_jsonstring,
                              watershed_segmentation)
from skimage import io, color, img_as_ubyte
import io
import cv2
import scipy.misc
import numpy as np
import datetime
import base64
from base64 import decodestring

#from dash import array_to_data_string

app = dash.Dash(__name__)
app.config['suppress_callback_exceptions']=True
static_image_route = '/static/'

# filename = 'F:\\ABEN\\Image Segmenation Tool\\static\\2004.jpg'
# filename2 = 'F:\\ABEN\\Image Segmenation Tool\\static\\2006.png'

# img = Image.open(filename)
#img2 = Image.open('F:\\ABEN\\Image Segmenation Tool\\static\\IMG_0202.jpg')
#img = array_to_data_url(img)

canvas_width = 500

columns = ['type', 'width', 'height', 'scaleX', 'strokeWidth', 'path']

app.layout = html.Div([
    html.Hr(),
    html.H1('Precision agriculture image Segmenation tool'),
    html.Hr(),
    html.H2('Upload RGB and NDVI images in the dropbox below'),
    html.Div([
      ##############  drag and drop  ################
      dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select RGB image')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
      dcc.Upload(
        id='upload-NDVIimage',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select NDVI image')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Button('Upload RGB', id='button'),
    html.Div([
        html.Div(id='output-image-upload'),
        html.Div(id='output-NDVIimage-upload'),
        ],style={'height':'100', 'width':'100'}),
    
##############################################
      # DashCanvas(id='annot-canvas',
      #          lineWidth=5,
      #          image_content = img,
      #          # filename=filename,
      #          width=canvas_width,
      #          ),
      ], style={'textAlign': 'center','display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 'width': '32.5%'}, className="five columns"),

    html.Div([
    html.Img(id='segmentation-img', width=100),
    html.Img(id='segmentation-NDVIimg', width=100),
    ], className="five columns", style={'height':'100', 'width':'100'}),
    ], style={'textAlign': 'center','background-color': 'rgb(45, 72, 115)','color': 'white'})
################################################################################################
################################################################################################
################################################################################################

def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        #html.Img(src=contents),
        html.Hr(),
        DashCanvas(id='annot-canvas',
               lineWidth=5,
               image_content = contents,
               # filename=filename,
               width=500,
               ),
    ])


@app.callback(Output('output-image-upload', 'children'),
              [Input('button', 'n_clicks')],
              [State('upload-image', 'contents'),
               State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])

def update_output_div(n_clicks, list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        #print(type(list_of_contents))
        children = parse_contents(list_of_contents, list_of_names, list_of_dates)
        #children = [ parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]
        return children
    #return input_value, n_clicks

# def update_output(list_of_contents, list_of_names, list_of_dates):
#     if list_of_contents is not None:
#         children = [ parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]
#         return children
##################################################################
def parse_contentsNDVI(contentsNDVI, filenameNDVI, dateNDVI):
    return html.Div([
        html.H5(filenameNDVI),
        html.H6(datetime.datetime.fromtimestamp(dateNDVI)),
        html.Img(src=contentsNDVI),
        html.Hr(),
    ])


@app.callback(Output('output-NDVIimage-upload', 'children'),
              [Input('upload-NDVIimage', 'contents')],
              [State('upload-NDVIimage', 'filename'),
               State('upload-NDVIimage', 'last_modified')])

def update_output_divNDVI(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        childrenNDVI = parse_contentsNDVI(list_of_contents, list_of_names, list_of_dates)
        return childrenNDVI
################################################################################################
################################################################################################
################################################################################################
@app.callback([Output('segmentation-img', 'src'),Output('segmentation-NDVIimg', 'src')],
              [Input('annot-canvas', 'json_data'),Input('upload-image', 'contents'),Input('upload-NDVIimage', 'contents')])
def segmentation(string, content, NDVIcontent):
    if string:
        #print(img)
        #mask = parse_jsonstring(string, io.imread(img, as_gray=True).shape)
        data = content.encode("utf8").split(b";base64,")[1]
        NDVIdata = NDVIcontent.encode("utf8").split(b";base64,")[1]
        #####################################
        img = io.BytesIO()
        imgNDVI = io.BytesIO()
        img.write(base64.b64decode(data))
        imgNDVI.write(base64.b64decode(NDVIdata))
        #print(type(img))
        img.seek(0)
        imgNDVI.seek(0)
        #i =  Image.open(img)
        i = np.asarray(bytearray(img.read()), dtype=np.uint8)
        i = cv2.imdecode(i, cv2.IMREAD_COLOR)
        iNDVI = np.asarray(bytearray(imgNDVI.read()), dtype=np.uint8)
        iNDVI = cv2.imdecode(iNDVI, cv2.IMREAD_COLOR)

        # iNDVI = np.asarray(i)
        # iNDVI.resize((i.shape[0],(i.shape[1]*i.shape[2])))
        #print(type(i))
        ########################################
        # with open(os.path.join('F:\\ABEN\\Image Segmenation Tool\\static\\', 'trial.png'), "wb") as fp:
        #     fp.write(base64.decodebytes(data))
        #img = Image.frombytes('RGB',(150,150),decodestring(data))
        #img = Image.open(io.BytesIO(base64.decodebytes(data)))
        #w, h = image.size
        mask = parse_jsonstring(string, (i.shape[0],i.shape[1]))
        # print(type(mask))
        #mask = cv2.UMat(mask)
        ret,thresh = cv2.threshold(np.array(mask, dtype=np.uint8), 0, 255, cv2.THRESH_BINARY)
        im_floodfill = thresh.copy()
        h, w = thresh.shape[:2]
        m = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, m, (0,0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = thresh | im_floodfill_inv
        #im_out = cv2.bitwise_and(io.imread(filename, as_gray=False), io.imread(filename, as_gray=False), mask=cv2.bitwise_not(im_out))
        ###NDVI = cv2.bitwise_and(io.imread(filename, as_gray=False), io.imread(filename, as_gray=False), mask=im_out)
        RGBimg = cv2.bitwise_and(i, i, mask=im_out)
        RGBimg = cv2.cvtColor(RGBimg, cv2.COLOR_BGR2RGB)

        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
        #cv2.drawContours(thresh,[cnts[0]], 0, (0,255,0), 30)
        (x,y,w,h) = cv2.boundingRect(cnts[0])
        ROIRGB = cv2.cvtColor(i[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        ROINDVI = cv2.cvtColor(iNDVI[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)

        # NDVIimg = cv2.bitwise_and(iNDVI[;;0], iNDVI[;;0], mask=im_out)
        # NDVIimg = cv2.cvtColor(NDVIimg, cv2.COLOR_BGR2RGB)
        ###image2 = io.imread(filename, as_gray=False)
        #image2.resize((NDVI.shape[0],(NDVI.shape[1]*NDVI.shape[2])))
        ###im_out2 = cv2.bitwise_and(image2, image2, mask=im_out)
        ###i = np.hstack((NDVI, im_out2))
        ###i = np.concatenate((NDVI, im_out2), axis=1)


        #numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
        #im_out = cv2.bitwise_and(im_out, im_out, mask=cv2.bitwise_not(io.imread(filename, as_gray=False)))

        # contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(np.array(mask, dtype=np.uint8), contours, -1, (255,255,255), thickness=-1)

        # cv2.imwrite('mask.png',mask.astype(np.uint8))
        
        #seg = watershed_segmentation(io.imread(filename, as_gray=True), mask)
        #src = color.label2rgb(seg, image=io.imread(filename, as_gray=True))
    else:
        raise PreventUpdate
    return array_to_data_url(img_as_ubyte(ROIRGB)), array_to_data_url(img_as_ubyte(ROINDVI)) 



if __name__ == '__main__':
    app.run_server(debug=True)


