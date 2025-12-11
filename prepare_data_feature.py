import os
import torch
from torchtext.legacy import data
import configs
from antlr4 import *
from JavaLexer import JavaLexer
import nltk
import ast
nltk.download('punkt')
from nltk.tokenize import word_tokenize
SEED = 3407
def code_tokenize(code):
    code=ast.literal_eval(code)
    code=code.decode('utf-8')
    code=code.strip()
    if code[0]=='*':
        code=code[1:]
    if code[-1]=='*':
        code=code+'/'
    try:
        lexer = JavaLexer(InputStream(code))
        token_stream = CommonTokenStream(lexer)
        token_stream.fill()
        tokens = token_stream.tokens[:-1]
        new_tokens=[]
        for token in tokens:
            if token.channel==0:
                if token.type==75:
                    new_tokens.extend(word_tokenize(token.text.lower()))
                else:
                    new_tokens.append(token.text.lower())
    except Exception as e:
        print(e)
        new_tokens=word_tokenize(code)
    if len(new_tokens) == 0:
        new_tokens = ['null']
    if len(new_tokens)>512:
        new_tokens=new_tokens[:512]
    return new_tokens
def field_tokenize(code):
    code=code.strip()
    if code[0]=='*':
        code=code[1:]
    if code[-1]=='*':
        code=code+'/'
    try:
        lexer = JavaLexer(InputStream(code))
        token_stream = CommonTokenStream(lexer)
        token_stream.fill()
        tokens = token_stream.tokens[:-1]
        new_tokens=[]
        for token in tokens:
            if token.channel==0:
                if token.type==75:
                    new_tokens.extend(word_tokenize(token.text.lower()))
                else:
                    new_tokens.append(token.text.lower())
    except Exception as e:
        print(e)
        new_tokens=word_tokenize(code)
    if len(new_tokens) == 0:
        new_tokens = ['null']
    if len(new_tokens)>256:
        new_tokens=new_tokens[:256]
    return new_tokens
def msg_tokenize(msg):
    tokens=word_tokenize(msg)
    if len(tokens)>32:
        tokens=tokens[:32]
    return tokens
def field_warning_tokenize(field):
    if field=='':
        return ['null']
    else:
        return [field.strip().lower()]
    
def ir_tokenize(binary):
    tokens=word_tokenize(binary)
    if len(tokens)>512:
        tokens=tokens[:512]
    return tokens
class CustomDataset(data.Dataset):
    def __init__(self, examples, fields, **kwargs):
        super(CustomDataset, self).__init__(examples, fields, **kwargs)

def prepare_data():
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    #javalang token
    CODE = data.Field(tokenize = code_tokenize,
                    include_lengths=True,batch_first=True)
    IRCODE = data.Field(tokenize = ir_tokenize,
                            include_lengths=True,batch_first=True)
    MSG = data.Field(tokenize = msg_tokenize,
                    include_lengths=True,batch_first=True)
    FIELD=data.Field(tokenize = field_tokenize,
                    include_lengths=True,batch_first=True)
    FIELDSWARNING = data.Field(tokenize = field_warning_tokenize,
                    batch_first=True)
    PRIORITY = data.Field(sequential=False, use_vocab=False,dtype=torch.long,batch_first=True)
    RANK = data.Field(sequential=False, use_vocab=False,dtype=torch.long,batch_first=True)      
    RULE = data.Field(sequential=False, use_vocab=False,dtype=torch.long,batch_first=True)
    CAT = data.Field(sequential=False, use_vocab=False,dtype=torch.long,batch_first=True)
    
    LABEL = data.LabelField(dtype = torch.float,batch_first=True)

    if not configs.save_dataset:
        fields = [("Code", CODE),("IRCode", IRCODE),("Msg", MSG),('Field',FIELD),("Cat", CAT),("Rule",RULE),("Priority",PRIORITY),("Rank",RANK),('FieldWarning',FIELDSWARNING),("Label",LABEL)]
        train_data, valid_data, test_data = data.TabularDataset.splits(
            path=configs.data_path,
            train=configs.train_file,
            validation=configs.valid_file,
            test=configs.test_file,
            format='csv',
            fields=fields,
            skip_header=True
        )
        MAX_VOCAB_SIZE = 100000
        CODE.build_vocab(train_data.Code,train_data.Field, max_size = MAX_VOCAB_SIZE)
        IRCODE.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        MSG.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        LABEL.build_vocab(train_data)
        FIELDSWARNING.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        FIELD.build_vocab(train_data.Code,train_data.Field, max_size = MAX_VOCAB_SIZE)
        torch.save(CODE, os.path.join(configs.save_path,'field_code.pth'))
        torch.save(MSG, os.path.join(configs.save_path,'field_msg.pth'))
        torch.save(IRCODE, os.path.join(configs.save_path,'field_ircode.pth'))
        torch.save(RULE, os.path.join(configs.save_path,'field_rule.pth'))
        torch.save(CAT, os.path.join(configs.save_path,'field_cat.pth'))
        torch.save(FIELD, os.path.join(configs.save_path,'field_field.pth'))
        torch.save(FIELDSWARNING, os.path.join(configs.save_path,'field_fieldwarning.pth'))
        torch.save(PRIORITY, os.path.join(configs.save_path,'field_priority.pth'))
        torch.save(RANK, os.path.join(configs.save_path,'field_rank.pth'))
        torch.save(LABEL, os.path.join(configs.save_path,'field_label.pth'))
        torch.save(train_data.examples, os.path.join(configs.save_path,'train_data.pth'))
        torch.save(valid_data.examples, os.path.join(configs.save_path,'valid_data.pth'))
        torch.save(test_data.examples, os.path.join(configs.save_path,'test_data.pth'))
    else:

        train_data_examples = torch.load(os.path.join(configs.save_path,'train_data.pth'))
        valid_data_examples = torch.load(os.path.join(configs.save_path,'valid_data.pth'))
        test_data_examples = torch.load(os.path.join(configs.save_path,'test_data.pth'))
        CODE=torch.load(os.path.join(configs.save_path,'field_code.pth'))
        IRCODE=torch.load(os.path.join(configs.save_path,'field_ircode.pth'))
        MSG=torch.load(os.path.join(configs.save_path,'field_msg.pth'))
        RULE=torch.load(os.path.join(configs.save_path,'field_rule.pth'))
        CAT=torch.load(os.path.join(configs.save_path,'field_cat.pth'))
        FIELD=torch.load(os.path.join(configs.save_path,'field_field.pth'))
        FIELDSWARNING=torch.load(os.path.join(configs.save_path,'field_fieldwarning.pth'))
        PRIORITY=torch.load(os.path.join(configs.save_path,'field_priority.pth'))
        RANK=torch.load(os.path.join(configs.save_path,'field_rank.pth'))
        LABEL=torch.load(os.path.join(configs.save_path,'field_label.pth'))
        fields = [("Code", CODE),("IRCode", IRCODE),("Msg", MSG),('Field',FIELD),("Cat", CAT),("Rule",RULE),("Priority",PRIORITY),("Rank",RANK),('FieldWarning',FIELDSWARNING),("Label",LABEL)]
        train_data = CustomDataset(train_data_examples, fields)
        valid_data = CustomDataset(valid_data_examples, fields)
        test_data = CustomDataset(test_data_examples, fields)
        MAX_VOCAB_SIZE = 100000
        CODE.build_vocab(train_data.Code,train_data.Field, max_size = MAX_VOCAB_SIZE)
        IRCODE.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        MSG.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        LABEL.build_vocab(train_data)
        FIELDSWARNING.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        FIELD.build_vocab(train_data.Code,train_data.Field, max_size = MAX_VOCAB_SIZE)
    device = configs.device
    BATCH_SIZE = configs.batch_size
    train_iter = data.BucketIterator(train_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.Code),
                                device=device, sort=True, sort_within_batch=True)
    valid_iter = data.BucketIterator(valid_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.Code),
                                device=device, sort=True, sort_within_batch=True)
    test_iter = data.BucketIterator(test_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.Code),
                                device=device, sort=True, sort_within_batch=True)


    return train_iter, valid_iter, test_iter
