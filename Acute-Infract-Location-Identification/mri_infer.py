####Inference Engine

from openvino.inference_engine import IENetwork, IEPlugin
import os
import time
import cv2
import argparse
import numpy as np
import tkinter as tk

root= tk.Tk()
canvas1 = tk.Canvas(root, width = 300, height = 300)
canvas1.pack()

target_names = {0: '0: Bilateral cerebellar hemispheres or 1: Bilateral cerebellar hemispheres and vermis',
1: '2: Bilateral frontal lobes',
2: '3: Bilateral occipital lobes or  12: Lacunar infarcts in bilateral occipital lobes',
3: '4: Brainstem or 5: Lacunar infarct in dorsal aspect of pons',
4: '6: Lacunar infarct in left parietal lobe or 25 :Left parietal lobe',
5: '7: Lacunar infarct in medulla oblongata on the left',
6: '8: Lacunar infarct in pons on the left',
7: '9: Lacunar infarct in posterior limb of left internal capsule',
8: '10: Lacunar infarct in right corona radiata or 33: Right corona radiata',
9: '11: Lacunar infarct in right putamen',
10:'13: Lacunar infarcts in left corona radiata',
11:'14: Lacunar infarcts in the right parietal lobe or 41: Right parietal lacunar infarct or 42: Right parietal lobe',
12:'15: Left centrum semi ovale and right parietal lobe ',
13:'16: Left cerebellar hemisphere or 17: Left cerebellar lacunar infarcts',
14:'18: Left frontal lobe or 20: Left fronto-parietal lobe or 22: Left insula',
15:'19: Left frontal lobe in precentral gyral location',
16:'21: Left fronto-temporo-parietal region',
17:'23: Left occipital and temporal lobes',
18:'24: Left occipital lobe',
19:'26: Left thalamic lacunar infarct',
20:'27: Medial part of right frontal and parietal lobes',
21:'28: Mid brain on right side',
22:'29: Pontine infarct on the right',
23:'30: Right anterior thalamic infarct',
24:'31: Right cerebellar hemisphere or 32: Right cerebellar hemisphere infarct',
25:'34: Right frontal lobe',
26:'35: Right fronto-parietal lobe or 37: Right ganglio-capsular region',
27:'36: Right fronto-parieto-temporo- occipital lobes',
28:'38: Right insula',
29:'39: Right lentiform nucleus',
30:'40: Right occipital lobe',
31:'43: Right temporal lobe',
32:'44: Right thalamus',
33:'45: Splenium of the corpus callosum'}

combined = {
0: '0: Bilateral cerebellar hemispheres or 1: Bilateral cerebellar hemispheres and vermis',
1: '2: Bilateral frontal lobes',
2: '3: Bilateral occipital lobes or  12: Lacunar infarcts in bilateral occipital lobes',
3: '4: Brainstem or 5: Lacunar infarct in dorsal aspect of pons',
4: '6: Lacunar infarct in left parietal lobe or 25 :Left parietal lobe',
5: '7: Lacunar infarct in medulla oblongata on the left',
6: '8: Lacunar infarct in pons on the left',
7: '9: Lacunar infarct in posterior limb of left internal capsule',
8: '10: Lacunar infarct in right corona radiata or 33: Right corona radiata',
9: '11: Lacunar infarct in right putamen',
10:'13: Lacunar infarcts in left corona radiata',
11:'14: Lacunar infarcts in the right parietal lobe or 41: Right parietal lacunar infarct or 42: Right parietal lobe',
12:'15: Left centrum semi ovale and right parietal lobe ',
13:'16: Left cerebellar hemisphere or 17: Left cerebellar lacunar infarcts',
14:'18: Left frontal lobe or 20: Left fronto-parietal lobe or 22: Left insula',
15:'19: Left frontal lobe in precentral gyral location',
16:'21: Left fronto-temporo-parietal region',
17:'23: Left occipital and temporal lobes',
18:'24: Left occipital lobe',
19:'26: Left thalamic lacunar infarct',
20:'27: Medial part of right frontal and parietal lobes',
21:'28: Mid brain on right side',
22:'29: Pontine infarct on the right',
23:'30: Right anterior thalamic infarct',
24:'31: Right cerebellar hemisphere or 32: Right cerebellar hemisphere infarct',
25:'34: Right frontal lobe',
26:'35: Right fronto-parietal lobe or 37: Right ganglio-capsular region',
27:'36: Right fronto-parieto-temporo- occipital lobes',
28:'38: Right insula',
29:'39: Right lentiform nucleus',
30:'40: Right occipital lobe',
31:'43: Right temporal lobe',
32:'44: Right thalamus',
33:'45: Splenium of the corpus callosum'}

ap = argparse.ArgumentParser()
ap.add_argument("-i","--input", help="path to input", required=True)
ap.add_argument("-d","--device", help="Device", required=True)
ap.add_argument("-m","--model", help="path to xml file", required=True)
args = ap.parse_args()
#reading the model----

model_xml = str(args.model)
model_bin = str(os.path.splitext(model_xml)[0])+ ".bin"

#Setup Devices----
plugin = IEPlugin(device=str(args.device)) #MYRIAD
net = IENetwork(model=model_xml, weights=model_bin)

#Allocating input and output blobs----
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))


#load model to plugin----
exec_net = plugin.load(network=net, num_requests=2)

#read and preprocess input images---
n, c, h, w = net.inputs[input_blob].shape
#print(n,c,h,w)

"""Note: compile OpenCV with JPEG file support enabled."""


#print(args.input)
image = cv2.imread(str(args.input))
image = image[100:850, 100:750]
if(image[:-1] != (w,h)):
	image = cv2.resize(image, (w, h))
image=image/255
image = image.transpose((2,0,1)) #as openvino expects in this format HWC to CHW
	
res = exec_net.infer(inputs={input_blob:image})
res = res[out_blob]
a = np.amax(res)
result = np.where(res == a)
out = combined[(result[1][0])+1]
print(res)
l=res
print("predicted 1 = ",target_names[l.argmax()],l.max())
l[0][l.argmax()]=0
print("predicted 2 = ",target_names[l.argmax()],l.max())
l[0][l.argmax()]=0
print("predicted 3 = ",target_names[l.argmax()],l.max())

label1 = tk.Label(root, text= out, fg='green', font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 150, window=label1)


root.mainloop()
