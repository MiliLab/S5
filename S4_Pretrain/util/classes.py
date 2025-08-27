# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Github repo: https://github.com/google-research/semivl/blob/main/datasets/classes.py


CLASSES = {'pascal': ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 
                      'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 
                      'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'],

           'pretrain': ['background', 'ship', 'storage_tank', 'baseball_diamond', 'tennis_court', 'basketball_court', 
                        'ground_Track_Field', 'bridge', 'large_Vehicle', 'small_Vehicle',
                        'helicopter', 'swimming_pool', 'roundabout', 'soccer_ball_field', 'plane', 'harbor'],

           'potsdam': ['Impervious_Surface', 'Building', 'Low_Vegetation', 'Tree', 'Car'],

           'vaihingen': ['Impervious_Surface', 'Building', 'Low_Vegetation', 'Tree', 'Car'],

           'loveda': ['Background ', 'Building', 'Road ', 'Water', 'Barren', 'Barren', 'Agriculture'],
           
           'cityscapes': ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                          'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                          'truck', 'bus', 'train', 'motorcycle', 'bicycle'],
           
           'coco': ['void', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
                    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 
                    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
                    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 
                    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 
                    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
                    'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 
                    'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 
                    'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
                    'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 
                    'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 
                    'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 
                    'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 
                    'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 
                    'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
                    'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table', 'tent',
                    'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other', 
                    'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
                    'window-blind', 'window-other', 'wood']
           }