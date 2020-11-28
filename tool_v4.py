#!/usr/bin/env python
# coding: utf-8
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
#from dash_extensions import Download
from zipfile import ZipFile
import datetime
from urllib.parse import quote as urlquote
from flask import Flask, send_from_directory

static_image_route = './static/'

if not os.path.exists(static_image_route):
    os.makedirs(static_image_route)

server = Flask(__name__)
app = dash.Dash(server=server)

@server.route("/download/<path:path>")
def download(path):
    return send_from_directory(static_image_route, path, as_attachment=True)

app.config['suppress_callback_exceptions']=True

canvas_width = 500

columns = ['type', 'width', 'height', 'scaleX', 'strokeWidth', 'path']

app.layout = html.Div([
    html.Hr(),
    html.H1('Semi-automatic Paired Dataset Creation Tool'),
    html.Hr(),
    html.H2('Upload Color and CIR Images in Below Boxes'),
    html.Div([
      dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select color image')
        ]),
        style={
            'width': '100%',
            'height': '200px',
            'lineHeight': '180px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
      html.Button('Upload RGB ', id='button', style={'display':'block'
                                    ,'position':'relative',
                                    'top': '45%','left': '45%',
                                    'font-size': '16px',
                                    'padding': '8px 12px',
                                    'border-radius': '4px',
                                    'text-align': 'center',
                                    'align':'center',
                                    'color':'black',
                                    'font-family':'Times New Roman',
                                    'textAlign':'center'}),
      dcc.Upload(
        id='upload-NDVIimage',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select CIR image')
        ]),
        style={
            'width': '100%',
            'height': '200px',
            'lineHeight': '180px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    
    html.Div([
        html.Div(id='output-image-upload'),
        html.Div(id='output-NDVIimage-upload'),
        ],style={'height':'100', 'width':'100'}),
          ], style={'textAlign': 'center','display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 'width': '32.5%','background-image': 'url(https://www.pexels.com/photo/scenic-view-of-agricultural-field-against-sky-during-sunset-325944/)'}, className="five columns"),

    html.Div([
    html.Img(id='segmentation-img', width=200),
    html.Img(id='segmentation-NDVIimg', width=200),
    html.Div(id='outputImages')
    ], className="five columns", style={'height':'100', 'width':'100'}),
    ], style={'textAlign': 'center','background-color': 'rgb(87, 95, 110)','color': 'white'})

def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        html.Hr(),
        DashCanvas(id='annot-canvas',
               lineWidth=5,
               image_content = contents,
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
        children = parse_contents(list_of_contents, list_of_names, list_of_dates)
        return children

def parse_contentsNDVI(contentsNDVI, filenameNDVI, dateNDVI):
    return html.Div([
        html.H5(filenameNDVI),
        html.H6(datetime.datetime.fromtimestamp(dateNDVI)),
        html.Div(html.Img(src=contentsNDVI, style={'height':'100%', 'width':'100%'})),
        #html.Img(src=contentsNDVI),
        html.Hr(),
    ],style={'height':'500', 'width':'500'})


@app.callback(Output('output-NDVIimage-upload', 'children'),
              [Input('upload-NDVIimage', 'contents')],
              [State('upload-NDVIimage', 'filename'),
               State('upload-NDVIimage', 'last_modified')])

def update_output_divNDVI(NDVIlist_of_contents, NDVIlist_of_names, NDVIlist_of_dates):
    if NDVIlist_of_contents is not None:
        childrenNDVI = parse_contentsNDVI(NDVIlist_of_contents, NDVIlist_of_names, NDVIlist_of_dates)
        return childrenNDVI


@app.callback([Output('segmentation-img', 'src'),Output('segmentation-NDVIimg', 'src'), Output('outputImages','children')],
              [Input('annot-canvas', 'json_data'),Input('upload-image', 'contents'),Input('upload-NDVIimage', 'contents')]) # receive data from canvas, input button and NDVI upload
def segmentation(string, content, NDVIcontent):
    global ROIRGB, ROINDVI
    if string:
        data = content.encode("utf8").split(b";base64,")[1]
        NDVIdata = NDVIcontent.encode("utf8").split(b";base64,")[1]
        img = io.BytesIO()
        imgNDVI = io.BytesIO()
        img.write(base64.b64decode(data))
        imgNDVI.write(base64.b64decode(NDVIdata))
        img.seek(0)
        imgNDVI.seek(0)
        i = np.asarray(bytearray(img.read()), dtype=np.uint8)
        i = cv2.imdecode(i, cv2.IMREAD_COLOR)
        iNDVI = np.asarray(bytearray(imgNDVI.read()), dtype=np.uint8)
        iNDVI = cv2.imdecode(iNDVI, cv2.IMREAD_COLOR)
        mask = parse_jsonstring(string, (i.shape[0],i.shape[1]))
        ret,thresh = cv2.threshold(np.array(mask, dtype=np.uint8), 0, 255, cv2.THRESH_BINARY)
        im_floodfill = thresh.copy()
        h, w = thresh.shape[:2]
        m = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, m, (0,0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = thresh | im_floodfill_inv
        RGBimg = cv2.bitwise_and(i, i, mask=im_out)
        RGBimg = cv2.cvtColor(RGBimg, cv2.COLOR_BGR2RGB)
        target_size = (RGBimg.shape[1],(RGBimg.shape[0]))
        iNDVI = cv2.resize(iNDVI, target_size)
        NDVIimg = cv2.bitwise_and(iNDVI, iNDVI, mask=im_out)
        NDVIimg = cv2.cvtColor(NDVIimg, cv2.COLOR_BGR2RGB)

        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)  # finds the largest selection, which is a limitation to multiple selection (will be changed in the future versions)
        (x,y,w,h) = cv2.boundingRect(cnts[0])
        ROIRGB = RGBimg[y:y+h, x:x+w]
        ROINDVI = NDVIimg[y:y+h, x:x+w]
        cv2.imwrite(static_image_route+'RGB_cropped.png', cv2.cvtColor(ROIRGB, cv2.COLOR_RGB2BGR ))
        cv2.imwrite(static_image_route+'CIR_cropped.png', cv2.cvtColor(ROINDVI, cv2.COLOR_RGB2BGR ))
        with ZipFile(static_image_route+'cropped.zip', 'w') as zipObj2:
            zipObj2.write(static_image_route+'RGB_cropped.png')
            zipObj2.write(static_image_route+'CIR_cropped.png')
            location = os.path.join(static_image_route,'cropped.zip')

    else:
        raise PreventUpdate
    return array_to_data_url(img_as_ubyte(ROIRGB)), array_to_data_url(img_as_ubyte(ROINDVI)), html.A(html.Button('Download',style={'display':'block'
                                    ,'position':'relative',
                                    'top': '45%','left': '45%',
                                    'font-size': '16px',
                                    'padding': '8px 12px',
                                    'border-radius': '4px',
                                    'text-align': 'center',
                                    'align':'center',
                                    'color':'black',
                                    'font-family':'Times New Roman',
                                    'textAlign':'center'}), href=location) #, html.Div([html.Button('Save Image', id='saveImg')])


def send_file(filename,mime_type=None):
    try:
        content = array_to_data_url(img_as_ubyte(ROIRGB))
        data = content.encode("utf8").split(b";base64,")[1]
        imgtowrite = io.BytesIO(base64.decodebytes(data))
        c = base64.b64encode(imgtowrite.read()).decode()
        return dict(content=c, filename=filename,mime_type=mime_type, base64=True)
    except Exception: pass

@app.callback(Output("download", "data"), [Input("DNWLD", "n_clicks")])

def func(n_clicks):
    return send_file(str(n_clicks)+".png")
    

if __name__ == '__main__':
    app.run_server(debug=True)


