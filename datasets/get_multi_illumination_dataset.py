from PIL import Image
import multilum as ml
import os
import numpy as np
import re

import argparse
TRAIN_BUILDINGS = ['kingston',
                   'main',
                   'summer',
                   'state',
                   'west']
VAL_BUILDINGS = ['joy',
                 'willow',
                 'memorial', 
                 'fulkerson', 
                 'elm', 
                 'marlborough']

ROOM_TO_IDX = {
    'basement': 0,
    'bathroom': 1,
    'bedroom': 2,
    'dining': 3,
    'kitchen': 4,
    'lab': 5,
    'living': 6,
    'lobby': 7,
    'office': 8
}

unknown_materials = [
    "I can't tell", 
    'More than one material', 
    'Not on list', 
    'no_consensus', 
    'unassigned', 
    "splitshape"
]

# From original dataset
material_to_palette =  {
    "unassigned": 0,
    "I can't tell": 1,
    "More than one material": 2,
    "Cardboard": 3,
    "Ceramic": 4,
    "Concrete": 5,
    "Cork/corkboard": 6,
    "Dirt": 7,
    "Fabric/cloth": 8,
    "Foliage": 9,
    "Food": 10,
    "Fur": 11,
    "Glass": 12,
    "Granite/marble": 13,
    "Laminate": 14,
    "Leather": 15,
    "Linoleum": 16,
    "Metal": 17,
    "Mirror": 18,
    "Not on list": 19,
    "Painted": 20,
    "Paper/tissue": 21,
    "Plastic - clear": 22,
    "Plastic - opaque": 23,
    "Rubber/latex": 24,
    "Sponge": 25,
    "Styrofoam": 26,
    "Tile": 27,
    "splitshape": 28,
    "Wallpaper": 29,
    "Wax": 30,
    "Wicker": 31,
    "Wood": 32,
    "Stone": 33,
    "Chalkboard/blackboard": 34,
    "Carpet/rug": 35,
    "Brick": 36,
    "Skin": 37,
    "Water": 38,
    "Hair": 39,
    "no_consensus": 40,
}

known_materials = sorted([k for k in material_to_palette.keys() if k not in unknown_materials])
reduced_material_to_palette = {k:i for i,k in enumerate(known_materials)}
reduced_material_to_palette['Other'] = len(known_materials)

palette_to_material = sorted(material_to_palette.keys(), key=lambda k: material_to_palette[k])
reduced_labels = { l: (l if l in reduced_material_to_palette.keys() else 'Other') for l in material_to_palette.keys()}

def f(x): 
    x = palette_to_material[x]
    x = reduced_labels[x]
    return reduced_material_to_palette[x]

def flat_for(a, f): 
    a = a.reshape(-1)
    for i,v in enumerate(a): 
        a[i] = f(v)

TEST_ROOMS = ['willow_basement39',
 'elm_storage20',
 'main_admin1',
 'willow_kitchen28',
 'kingston_storage28',
 'willow_living16',
 'willow_basement42',
 'willow_kitchen14',
 '14n_office6',
 'main_drylab25',
 'fulkerson_revis_dining4',
 'elm_1floor_bathroom7',
 'elm_storage22',
 'summer_living12',
 'elm_revis_kitchen4',
 'kingston_dining8',
 'willow_kitchen10',
 'marlborough_kitchen9',
 'main_experiment38',
 'elm_revis_living10',
 'elm_basebath9',
 'main_experiment100',
 'summer_bedroom19',
 'fulkerson_revis_living4',
 'main_experiment104',
 'elm_1floor_bedroom9',
 'marlborough_living16',
 'willow_bathroom4',
 'west_study19',
 'main_experiment28',
 'elm_2floor_bedroom_revisit2',
 'elm_2floor_bathroom8',
 '14n_office18',
 'joy_bathroom1',
 'fulkerson_revis_dining5',
 'elm_2floor_bedroom3',
 'joy_bedroom1',
 '32-d414-revisit1',
 'fulkerson_revis_dining8',
 'fulkerson_revis_dining6',
 'joy_hallway6',
 'joy_hallway3',
 'joy_hallway1',
 'willow_atrium4',
 'joy_hallway5']

def download_scenes_and_materials(mode, directory, val, mip): 
    ml.set_datapath(directory)

    if mode == 'train': 
        label='dataset-train'
        scenes=ml.train_scenes()
        if val == "drylab": 
            scenes = [s for s  in sorted(ml.train_scenes()) if 'main_drylab' not in s.name]
        else:
            raise ValueError
    elif mode == 'val': 
        label='dataset-val'
        scenes=ml.train_scenes()
        if val == "drylab": 
            scenes = [s for s  in sorted(ml.train_scenes()) if 'main_drylab' in s.name]
        else: 
            raise ValueError
    else:
        label='dataset-test'
        scenes=ml.test_scenes()

    for j,i in enumerate(range(0,len(scenes),200)): 
        print(j,i)
        scene_subset = scenes[i:min(i+200, len(scenes))]

        I = ml.query_images(scene_subset, mip=mip)
        np.save(os.path.join(directory, f'{label}-mip-{mip}-batch-{j}.npy'), I)
        
        # query MIP2 images and downsample with nearest neighbors due to off 
        # by one bug in the dimension of the material map
        if mip == 5: 
            M2 = ml.query_materials(scene_subset, mip=2)
            M5 = []
            for m in M2: 
                M5.append(np.array(Image.fromarray(m).resize((187, 125), Image.NEAREST)).reshape(1,125,187))
            M5 = np.concatenate(M5,0)
            flat_for(M5, f)
            np.save(os.path.join(directory, f'{label}-materials-mip-{mip}-batch-{j}.npy'), M5)
        else: 
            M = ml.query_materials(scene_subset, mip=mip)
            flat_for(M, f)
            np.save(os.path.join(directory, f'{label}-materials-mip-{mip}-batch-{j}.npy'), M)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'test', 'val', 'all'], default='all')
    parser.add_argument("--dir", default="./dataset")
    parser.add_argument("--validation", choices=['building', 'drylab', 'segmentation', 'rooms'], default="drylab") 
    parser.add_argument("--mip", default=5, type=int)
    args = parser.parse_args() 

    if args.mode == 'all': 
        for mode in ['train', 'test', 'val']: 
            download_scenes_and_materials(mode, args.dir, args.validation,
                                            args.mip)
    else: 
        download_scenes_and_materials(args.mode, args.dir, args.validation,
                                            args.mip)
