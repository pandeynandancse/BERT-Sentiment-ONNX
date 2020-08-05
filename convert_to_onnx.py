#convert any trained pytorch model to ONNX and serve it using flask
import torch.onnx
from torch import nn

import config
import dataset

from model import BERTBaseUncased


if __name__ == '__main__':
	device='cuda'
	review = ['this is an amzaing place']

	dataset = BERTDataset(
			review = review,target=[0]
		)
	model = BERTBaseUncased()

	#model = nn.DataParallel(model) #if you have used model as DataParallel-model then inside torch.onnx.export use 'model.module' instead of model
	#====>>> question is from where 'module' arises ??=>>>> print model(uncomment line 25) here .You will output as Dataparallel({module:BERTBaseUncased())
	# ====> that means 'module' is key of 'BERTBaseUncased' model's value
	model.load_state_dict(torch.load(config.MODEL_PATH))
	model.eval()
	#print(model)

	ids = dataset[0]['ids'].unsqueeze(0)
	mask = dataset[0]['mask'].unsqueeze(0)
	token_type_ids = dataset[0][''token_type_ids].unsqueeze(0)

	torch.onnx.export(
			model,
			#model.module  [===>> if dataparallel-model ==>> see above commented line 20] 

			#we have 3 inputs so  name them here -- ordering is important 
			(ids,mask,token_type_ids),

			#export it to model.onnx
			'model.onnx',

			#in same order as we wrote inputs
			input_names = ['ids' 'mask', 'token_type_ids'],

			#anything that you want
			output_names = ['output']

			#we know that batch size is dynamic as axis of batch size is index 0 so ==>>> {0:"batch_size"} ==>> for each inputs and outputs
			dynamic_axes = {
				'ids':{0:"batch_size"},
				'mask':{0:"batch_size"},
				'token_type_ids':{0:"batch_size"},
				'output':{0:"batch_size"},
			}
		)