import os
from mmseg.datasets import DATASETS
from mmseg.datasets import PascalVOCDataset as _PascalVOC
from PIL import Image

from torchvision.transforms.functional import to_tensor, resize, InterpolationMode

class Wrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        a = self.dataset[index]
        img = a['img'][...,[2,1,0]]
        mask = Image.fromarray(a['gt_semantic_seg'])
        img = Image.fromarray(img)
        return img, mask

    def __len__(self):
        return len(self.dataset)

@DATASETS.register_module(force=True)
class PascalVOCDataset(_PascalVOC):
    CLASSES = (
        "__background__",
        "airplane",  # "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "table", # diningtable,
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "monitor", # "tvmonitor",
    )

    PALETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
               [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

    CLASS_TO_PROMPT = {
        "__background__": "background",
        "airplane": "airplane",
        "bicycle": "bicycle",
        "bird": "bird",
        "boat": "boat",
        "bottle": "bottle",
        "bus": "bus",
        "car": "car",
        "cat": "cat",
        "chair": "chair",
        "cow": "cow",
        "table": "table", # <-- 
        "dog": "dog",
        "horse": "horse",
        "motorbike": "motorbike",
        "person": "person",
        "pottedplant": "potted plant", # <-- 
        "sheep": "sheep",
        "sofa": "sofa",
        "train": "train",
        "monitor": "monitor" # <-- 
    }


def prep_sample(example, size=(512,512)):
    img, mask = example
    img = to_tensor(resize(img, size, interpolation=InterpolationMode.LANCZOS))
    # mask = torch.from_numpy(np.array((resize(mask, (512, 512), interpolation=InterpolationMode.NEAREST)))).long().unsqueeze(0)
    mask = resize(mask, size, interpolation=InterpolationMode.NEAREST)
    return img, mask


def _get_voc():
    dataset = PascalVOCDataset(
        data_root='data/PascalVOC/VOCdevkit/VOC2012',
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
        ],
    )
    return dataset

CLASS_TO_PROMPT_OVERLAY = {
    'tank': 'liquid tank',
    'blind': 'window blinds',
    'hood': 'cooker hood',
    'ashcan': 'wastebin',
    'case': 'shop case',
    'keyboard': 'computer keyboard',
    'apparel': 'clothing',
}

def get_dataset(dataset, apply_overlay=False):
    if dataset == 'voc':
        r = Wrapper(_get_voc()), PascalVOCDataset.CLASSES, PascalVOCDataset.CLASS_TO_PROMPT
    else:
        raise NotImplementedError()
    if apply_overlay:
        new_class_to_prompt = {}
        cls2prompt = r[2]
        for k, v in cls2prompt.items():
            if k in CLASS_TO_PROMPT_OVERLAY:
                new_class_to_prompt[k] = CLASS_TO_PROMPT_OVERLAY[k]
                print(f"Overriding for {k} with <{v}> with <{CLASS_TO_PROMPT_OVERLAY[k]}>")
            else:
                new_class_to_prompt[k] = v
        r = r[0], r[1], new_class_to_prompt
    return r

def get_data_classes(dataset, apply_overlay=False):
    if dataset == 'voc':
        r = PascalVOCDataset.CLASSES, PascalVOCDataset.CLASS_TO_PROMPT
    else:
        raise NotImplementedError()
    if apply_overlay:
        new_class_to_prompt = {}
        cls2prompt = r[1]
        for k, v in cls2prompt.items():
            if k in CLASS_TO_PROMPT_OVERLAY:
                new_class_to_prompt[k] = CLASS_TO_PROMPT_OVERLAY[k]
                print(f"Overriding for {k} with <{v}> with <{CLASS_TO_PROMPT_OVERLAY[k]}>")
            else:
                new_class_to_prompt[k] = v
        r = r[0], new_class_to_prompt
    return r


CHAT_GPT_REPLY_LONGER = """
background: stuff
__background__: stuff
airplane: thing
bag: thing
bed: thing
bedclothes: stuff
bench: thing
bicycle: thing
bird: thing
boat: thing
book: thing
bottle: thing
building: thing
bus: thing
cabinet: thing
car: thing
cat: thing
ceiling: stuff
chair: thing
cloth: stuff
computer: thing
cow: thing
cup: thing
curtain: stuff
dog: thing
door: thing
fence: stuff
floor: stuff
flower: thing
food: thing
grass: stuff
ground: stuff
horse: thing
keyboard: thing
light: thing
motorbike: thing
mountain: stuff
mouse: thing
person: thing
plate: thing
platform: stuff
plant: thing
road: stuff
rock: stuff
sheep: thing
shelves: thing
sidewalk: stuff
sign: thing
sky: stuff
snow: stuff
sofa: thing
table: thing
track: stuff
train: thing
tree: thing
truck: thing
monitor: thing
wall: stuff
water: stuff
window: thing
wood: stuff
windowpane: thing
earth: thing
painting: thing
shelf: thing
house: thing
sea: thing
mirror: thing
rug: thing
field: thing
armchair: thing
seat: thing
desk: thing
wardrobe: thing
lamp: thing
bathtub: thing
railing: thing
cushion: thing
base: thing
box: thing
column: thing
signboard: thing
chest of drawers: thing
counter: thing
sand: thing
sink: thing
skyscraper: thing
fireplace: thing
refrigerator: thing
grandstand: thing
path: thing
stairs: thing
runway: thing
case: thing
pool table: thing
pillow: thing
screen door: thing
stairway: thing
river: thing
bridge: thing
bookcase: thing
blind: thing
coffee table: thing
toilet: thing
hill: thing
countertop: thing
stove: thing
palm: thing
kitchen island: thing
swivel chair: thing
bar: thing
arcade machine: thing
hovel: thing
towel: thing
tower: thing
chandelier: thing
awning: thing
streetlight: thing
booth: thing
television receiver: thing
dirt track: thing
apparel: thing
pole: thing
land: thing
bannister: thing
escalator: thing
ottoman: thing
buffet: thing
poster: thing
stage: thing
van: thing
ship: thing
fountain: thing
conveyer belt: thing
canopy: thing
washer: thing
plaything: thing
swimming pool: thing
stool: thing
barrel: thing
basket: thing
waterfall: thing
tent: thing
minibike: thing
cradle: thing
oven: thing
ball: thing
step: stuff
tank: thing
trade name: stuff
microwave: thing
pot: thing
animal: thing
lake: stuff
dishwasher: thing
screen: thing
blanket: stuff
sculpture: thing
hood: thing
sconce: thing
vase: thing
traffic light: thing
tray: stuff
ashcan: thing
fan: thing
pier: thing
crt screen: thing
bulletin board: thing
shower: thing
radiator: thing
glass: stuff
clock: thing
flag: thing
""" + """
"""
IS_STUFF = {l.strip().split(':')[0].strip():l.strip().split(':')[-1].strip()=='stuff'  for l in CHAT_GPT_REPLY_LONGER.strip().splitlines()}
