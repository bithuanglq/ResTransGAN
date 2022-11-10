import SimpleITK as sitk
import numpy as np
import glob
import json
from tqdm import tqdm
import os

'''
view the .mha ==> save to .json ==> preprocess .mha to .npy 
'''

if __name__=='__main__':
    if False:
        '''
             view the .mha
        '''
        mha_path = 'BraTS2015/BRATS2015_Training/HGG/brats_2013_pat0001_1/VSD.Brain_3more.XX.O.OT.54517/VSD.Brain_3more.XX.O.OT.54517.mha'
        img = sitk.ReadImage(mha_path)
        arr = sitk.GetArrayFromImage(img)
        origin = img.GetOrigin()
        # print(arr.shape, np.unique(arr))        # (155, 240, 240) [0 1 2 3 4]
        print(origin)       # (0.0, -239.0, 0.0)   or   (0.0, 0.0, 0.0) 
        '''
        when evaluation:
        e_WT = label1 + label2 + label3 + label4
        e_TC = label1 + label3 + label4
        e_ET = label4 
        '''


    if False:
        '''
            save to .json
        '''
        dirpath = 'BraTS2015/Testing/data.json'

        # Seg = glob.glob('BraTS2015/*Train*/*GG/*/*/*OT.*mha')  # Ground Truth
        T1 = glob.glob('BraTS2015/*Test*/*GG/*/*/*T1.*mha') 
        T1c = glob.glob('BraTS2015/*Test*/*GG/*/*/*T1c.*mha') 
        T2 = glob.glob('BraTS2015/*Test*/*GG/*/*/*T2.*mha') 
        Flair = glob.glob('BraTS2015/*Test*/*GG/*/*/*Flair.*mha') 
        
        # idx=100
        # print(idx, Seg[idx], T1[idx], T2[idx], T1c[idx], Flair[idx])

        d = dict()
        cnt = 0
        for idx in tqdm(range(len(T1))):                           # 总共274个 训练集
            # seg, t1, t1c, t2, flair = Seg[idx], T1[idx], T1c[idx], T2[idx], Flair[idx]
            # prefix = seg.split('/')[2]
            t1, t1c, t2, flair = T1[idx], T1c[idx], T2[idx], Flair[idx]
            prefix = t1.split('/')[2]
            if (t1.split('/')[2] != prefix) or (t1c.split('/')[2] != prefix) or (t2.split('/')[2] != prefix) or (flair.split('/')[2] != prefix):
                print(prefix,'\tERROR!')
                break
            tmp_d = dict()
            # tmp_d['seg'] = seg
            tmp_d['t1'] = t1
            tmp_d['t2'] = t2
            tmp_d['t1c'] = t1c
            tmp_d['flair'] = flair
            # seg = sitk.ReadImage(seg)
            seg = sitk.ReadImage(t1)
            tmp_d['spacing'] = (seg.GetSpacing())
            tmp_d['size'] = (seg.GetSize())
            tmp_d['origin'] = (seg.GetOrigin())
            tmp_d['direction'] = (seg.GetDirection())
            # seg_arr = sitk.GetArrayFromImage(seg)
            # tmp_d['labels'] = (np.unique(seg_arr).tolist())
            # if len(tmp_d['labels']) != 5:
            #     print(tmp_d['seg'],'\t',tmp_d['labels'])        # 存在:[0,2,3,4]    [0,1,2,3]   [0,2,3]  其他都素[0,1,2,3,4]

            d[cnt] = tmp_d
            cnt += 1


        # Using a JSON string
        json_string = json.dumps(d, indent=4)
        with open(dirpath, 'w') as outfile:
            outfile.write(json_string)

    if False:
        '''
            from mha to npy
        '''
        json_path = 'BraTS2015/Testing/data.json'
        savepath = 'preprocessed_data_2015/validation'


        with open(json_path) as json_file:
            data = json.load(json_file)
            length = len(data.keys())

            for index in tqdm(range(length)):
                # image
                x, y = 40, 24
                t1 = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(data[str(index)]['t1'])), (2,1,0))[x:(x+160), y:(y+192), :]    # (160, 192, 155)
                t1c = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(data[str(index)]['t1c'])), (2,1,0))[x:(x+160), y:(y+192), :] 
                t2 = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(data[str(index)]['t2'])), (2,1,0))[x:(x+160), y:(y+192), :] 
                flair = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(data[str(index)]['flair'])), (2,1,0))[x:(x+160), y:(y+192), :] 

                t1 = np.append(t1, np.ones((160, 192, 5))*np.min(t1), axis=2)     # (160, 192, 160)
                t1c = np.append(t1c, np.zeros((160, 192, 5))*np.min(t1c), axis=2)
                t2 = np.append(t2, np.zeros((160, 192, 5))*np.min(t2), axis=2)
                flair = np.append(flair, np.zeros((160, 192, 5))*np.min(flair), axis=2)

                image = np.append(t1[None], t1c[None], axis=0)
                image = np.append(image, t2[None], axis=0)
                image = np.append(image, flair[None], axis=0)   # (4, 160, 192, 160)


                # image normalized to (-1, 1)
                image_normalized = image.copy()
                for c in range(image.shape[0]):
                    modality = image_normalized[c,:,:,:]
                    mu, std = modality.mean(), modality.std()
                    modality = (modality - mu) / std
                    maxa, mina = np.max(modality), np.min(modality)
                    image_normalized[c,:,:,:] = (modality - mina) / (maxa - mina) * 2 + (-1)            # values:(-1,1)

                # combine image and image_normalized
                image = np.append(image, image_normalized, axis=0)      # (4+4, 160, 192, 160)

                # # mask
                # mask = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(data[str(index)]['seg'])), (2,1,0))[x:(x+160), y:(y+192), :] 
                # mask = np.append(mask, np.zeros((160, 192, 5)), axis=2)


                # NE = np.where(mask==1, 1, 0)[None]
                # ED = np.where(mask==2, 1, 0)[None]
                # NET = np.where(mask==3, 1, 0)[None]
                # EC = np.where(mask==4, 1, 0)[None]
                # mask_onehot = np.append(NE, ED, axis=0)
                # mask_onehot = np.append(mask_onehot, NET, axis=0)
                # mask_onehot = np.append(mask_onehot, EC, axis=0)                                  # (4, 160, 192, 160) NE, ED, NET, EC
                # ET = np.where(mask==4, 1, 0)[None]              # ET = label4
                # TC = np.where((mask==4) | (mask==3) | (mask==1), 1, 0)[None]    # TC = label1 + 3 + 4
                # WT = np.where((mask==4) | (mask==3) | (mask==2) | (mask==1), 1, 0)[None]    # WT = label1 + 2 + 3 + 4
                # mask_onehot = np.append(mask_onehot, ET, axis=0)
                # mask_onehot = np.append(mask_onehot, TC, axis=0)
                # mask_onehot = np.append(mask_onehot, WT, axis=0) 

                # npy = np.append(image, mask_onehot, axis=0)     #(8+4+3, 160, 192, 160)
                # if npy.shape != (8+7, 160, 192, 160):
                #     print(data[str(index)]['t1'], '\t', npy.shape)
                # np.save(os.path.join(savepath, 'VSD.' + data[str(index)]['flair'].split('/')[-3] + '.' + data[str(index)]['flair'].split('/')[4].split('.')[-1]), npy)


                if image.shape != (8, 160, 192, 160):
                    print(data[str(index)]['t1'], '\t', image.shape)
                np.save(os.path.join(savepath, 'VSD.' + data[str(index)]['flair'].split('/')[-3] + '.' + data[str(index)]['flair'].split('/')[4].split('.')[-1]), image)
