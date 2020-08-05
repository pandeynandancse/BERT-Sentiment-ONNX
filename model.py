import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
	
	#bert base uncased has 768 outputs 
        # 1 becoz binary calssification problem
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
	#a lot of things can be done here for best model but let keep it simple ===>> see bert documentation
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output
