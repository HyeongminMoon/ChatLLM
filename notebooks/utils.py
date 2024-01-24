qna_list = [
    "/data/llm_datasets/custom/vicuna_format/gpt_evol_1.3k-vicuna.json",
    "/data/llm_datasets/custom/vicuna_format/koalpaca_v1.1-vicuna.json",
    "/data/llm_datasets/custom/deduped/alpaca-gpt4-korean_dedup/",
    "/data/llm_datasets/custom/vicuna_format/korquad-chat-vicuna.json",
    "/data/llm_datasets/custom/vicuna_format/wizardlm_orca_vicuna.json",
    "/data/llm_datasets/sharegpt_gpt4/sharegpt_gpt4.jsonl",
    "/data/llm_datasets/custom/vicuna_format/sharegpt_V3_format_others.json",
    "/data/llm_datasets/custom/deduped/sharegpt_V3_format_ko_selected.json",
    "/data/llm_datasets/custom/vicuna_format/lima_vicuna_format_ko.json",
]

correction_list = [
    "/data/llm_datasets/custom/vicuna_format/KoreaSpellingCorrection/"
]

summary_list = [
    "/data/llm_datasets/custom/deduped/aihub_summary_data_tech_dedup/",
    "/data/llm_datasets/aihub_summary_data/도서/",
    "/data/llm_datasets/aihub_summary_data/법률/",
    "/data/llm_datasets/custom/deduped/naver-news-summarization-ko-vicuna_dedup/",
    
]

translation_list = [
    "/data/llm_datasets/custom/vicuna_format/sharegpt_V3_format_translation(enko).json",
    "/data/llm_datasets/custom/vicuna_format/sharegpt_V3_format_translation(koen).json",
]


dataset_list = qna_list + correction_list + summary_list + translation_list

# dedup2
qna_list = [
    "/data/llm_datasets/custom/vicuna_format/gpt_evol_1.3k-vicuna.json",
    "/data/llm_datasets/custom/vicuna_format/koalpaca_v1.1-vicuna.json",
    "/data/llm_datasets/custom/deduped2/alpaca-gpt4-korean_dedup2.json",
    "/data/llm_datasets/custom/vicuna_format/korquad-chat-vicuna.json",
    "/data/llm_datasets/custom/deduped2/wizardlm_orca_vicuna_dedup2.json",
    "/data/llm_datasets/sharegpt_gpt4/sharegpt_gpt4.jsonl",#
    "/data/llm_datasets/custom/vicuna_format/sharegpt_V3_format_others.json",#
    "/data/llm_datasets/custom/deduped2/sharegpt_V3_format_ko_selected_dedup2.json",
    "/data/llm_datasets/custom/deduped2/lima_vicuna_format_ko.json",
]

# correction_list = [
#     "/data/llm_datasets/custom/deduped2/KoreaSpellingCorrection-10000.json",
# ]

summary_list = [
    "/data/llm_datasets/custom/deduped2/aihub_summary_data_tech_dedup-5000.json",
    "/data/llm_datasets/custom/deduped2/aihub_summary_data_book-5000.json",
    "/data/llm_datasets/custom/deduped2/aihub_summary_data_law-5000.json",
    "/data/llm_datasets/custom/deduped2/naver-news-summarization-ko-vicuna_dedup-5000.json",
    
]

translation_list = [
    "/data/llm_datasets/custom/deduped2/sharegpt_V3_format_translation(enko)-10000.json",
    "/data/llm_datasets/custom/deduped2/sharegpt_V3_format_translation(koen)-10000.json",
]


dataset_list = qna_list + summary_list + translation_list

# refine
qna_list = [
    "/data/llm_datasets/custom/vicuna_format/gpt_evol_1.3k-vicuna.json",
    "/data/llm_datasets/custom/vicuna_format/koalpaca_v1.1-vicuna.json",
    "/data/llm_datasets/custom/refined/alpaca-gpt4-korean_dedup2.json",
    "/data/llm_datasets/custom/vicuna_format/korquad-chat-vicuna.json",
    "/data/llm_datasets/custom/refined/wizardlm_orca_vicuna_dedup2.json",
    "/data/llm_datasets/custom/vicuna_format/sharegpt_gpt4.json",#
    "/data/llm_datasets/custom/vicuna_format/sharegpt_V3_format_others.json",#
    "/data/llm_datasets/custom/refined/sharegpt_V3_format_ko_selected_dedup2.json",
    "/data/llm_datasets/custom/refined/lima_vicuna_format_ko.json",
]

# correction_list = [
#     "/data/llm_datasets/custom/deduped2/KoreaSpellingCorrection-10000.json",
# ]

summary_list = [
    "/data/llm_datasets/custom/deduped2/aihub_summary_data_tech_dedup-5000.json",
    "/data/llm_datasets/custom/deduped2/aihub_summary_data_book-5000.json",
    "/data/llm_datasets/custom/deduped2/aihub_summary_data_law-5000.json",
    "/data/llm_datasets/custom/deduped2/naver-news-summarization-ko-vicuna_dedup-5000.json",
    
]

translation_list = [
    "/data/llm_datasets/custom/deduped2/sharegpt_V3_format_translation(enko)-10000.json",
    "/data/llm_datasets/custom/deduped2/sharegpt_V3_format_translation(koen)-10000.json",
]


dataset_list = qna_list + summary_list + translation_list

# dpo v2
dpo_list = [
    "/data/llm_datasets/ultrafeedback_binarized/data/train_prefs-00000-of-00001-17309c769bfe5733.parquet",
    "/data/llm_datasets/orca_dpo_pairs/",
    "/data/llm_datasets/distilabel-math-preference-dpo/data/",
]

dpo_list2 = [
    "/data/llm_datasets/custom/kodpo/untranslated/ultrafeedback_binarized.json",
    "/data/llm_datasets/custom/kodpo/untranslated/orca_dpo_pairs.json",
    "/data/llm_datasets/custom/kodpo/untranslated/distilabel-math-preference-dpo.json",
]

dpo_list3 = [
    "/data/llm_datasets/custom/kodpo/translated/ko_ultrafeedback_binarized.json",
    "/data/llm_datasets/custom/kodpo/translated/ko_orca_dpo_pairs.json",
    "/data/llm_datasets/custom/kodpo/translated/ko_distilabel-math-preference-dpo.json",
]

# dedup
dpo_list4 = [
    '/data/llm_datasets/custom/kodpo/deduped/ko_ultrafeedback_binarized.json',
    "/data/llm_datasets/custom/kodpo/translated/ko_orca_dpo_pairs.json",
    "/data/llm_datasets/custom/kodpo/translated/ko_distilabel-math-preference-dpo.json",
]

# refine codeblock
dpo_list5 = [
    '/data/llm_datasets/custom/kodpo/refined/ko_ultrafeedback_binarized.json',
    "/data/llm_datasets/custom/kodpo/translated/ko_orca_dpo_pairs.json",
    "/data/llm_datasets/custom/kodpo/translated/ko_distilabel-math-preference-dpo.json",
]

file_paths = glob.glob("/workspaces/data/llm_datasets/aihub/*[!tar|!sh]")

dataset_dict = {}
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    
    new_dataset = []
    idx = 0

    paths = glob.glob(os.path.join(file_path, '*.json'))
    
    for path in paths:
        with open(path, "r") as f:
            json_data = json.load(f)

        context_info = json_data['dataset']['context_info']
        for context_data in context_info:
            context = context_data['context']
            summary = context_data['summary']

            data_row = {
                'id': f"{file_name}_{idx}",
                'task': 'summarization',
                'conversations': [
                                    {'from': 'human', 'value': context},
                                    {'from': 'gpt', 'value': summary},
                                 ],
            }
            new_dataset.append(data_row)
            idx += 1
        
    print(f"file_name:{file_name} idx:{idx}", '\t\t\t\t\t\t', end='\r')
    dataset_dict[file_name] = new_dataset
    
    
train_dataset_list = dataset_dict['TL_EE_train'] + dataset_dict['TL_LA_train'] + dataset_dict['TL_ED_train'] + dataset_dict['TL_NA_train']
eval_dataset_list = dataset_dict['TL_EE_val'] + dataset_dict['TL_LA_val'] + dataset_dict['TL_ED_val'] + dataset_dict['TL_NA_val']
with open("/workspaces/data/llm_datasets/aihub_summary_data/train.json", "w") as json_file:
    json.dump(train_dataset_list, json_file)
with open("/workspaces/data/llm_datasets/aihub_summary_data/test.json", "w") as json_file:
    json.dump(eval_dataset_list, json_file)
dataset = load_dataset("/workspaces/data/llm_datasets/aihub_summary_data")
file_paths = glob.glob("/workspaces/data/llm_datasets/aihub/*summary*[!tar|!sh]")

dataset_dict = {}
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    
    new_dataset = []
    idx = 0

    paths = glob.glob(os.path.join(file_path, '*.json'))
    
    for path in paths:
        with open(path, "r") as f:
            json_data = json.load(f)

        documents = json_data['documents']
        for document in documents:
            text = document['text']
            abstractive = document['abstractive']

            summary = abstractive[0]
            context = []
            for _text in text:
                _context = ' '.join([_index_text['sentence'] for _index_text in _text])
                context.append(_context)
            context = '\n'.join(context)
            
            data_row = {
                'id': f"{file_name}_{idx}",
                'task': 'summarization',
                'conversations': [
                                    {'from': 'human', 'value': context},
                                    {'from': 'gpt', 'value': summary},
                                 ],
            }
            new_dataset.append(data_row)
            idx += 1
        
        print(f"file_name:{file_name} idx:{idx}", '\t\t\t\t\t\t', end='\r')
    dataset_dict[file_name] = new_dataset
for key in dataset_dict.keys():
    print(key, len(dataset_dict[key]))
train_dataset_list = dataset_dict['summary_law_train']
eval_dataset_list = dataset_dict['summary_law_val']
with open("/workspaces/data/llm_datasets/aihub_summary_data/법률/train.json", "w") as json_file:
    json.dump(train_dataset_list, json_file)
with open("/workspaces/data/llm_datasets/aihub_summary_data/법률/test.json", "w") as json_file:
    json.dump(eval_dataset_list, json_file)
file_paths = glob.glob("/workspaces/data/llm_datasets/aihub/*summary_book*[!tar|!sh]")

dataset_dict = {}
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    
    new_dataset = []
    idx = 0

    paths = glob.glob(os.path.join(file_path, '**/*.json'))
    
    for path in paths:
        with open(path, "r") as f:
            json_data = json.load(f)

        context = json_data['passage']
        summary = json_data['summary']
        
        data_row = {
            'id': f"{file_name}_{idx}",
            'task': 'summarization',
            'conversations': [
                                {'from': 'human', 'value': context},
                                {'from': 'gpt', 'value': summary},
                             ],
        }
        new_dataset.append(data_row)
        idx += 1
        
        print(f"file_name:{file_name} idx:{idx}", '\t\t\t\t\t\t', end='\r')
    dataset_dict[file_name] = new_dataset
for key in dataset_dict.keys():
    print(key, len(dataset_dict[key]))
train_dataset_list = dataset_dict['summary_book_train']
eval_dataset_list = dataset_dict['summary_book_val']
with open("/workspaces/data/llm_datasets/aihub_summary_data/도서/train.json", "w") as json_file:
    json.dump(train_dataset_list, json_file)
with open("/workspaces/data/llm_datasets/aihub_summary_data/도서/test.json", "w") as json_file:
    json.dump(eval_dataset_list, json_file)
dataset = load_dataset("/workspaces/data/llm_datasets/aihub_summary_data/도서")
dataset
file_paths = glob.glob("/workspaces/data/llm_datasets/aihub/*VL*[!tar|!sh]")

dataset_dict = {}
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    
    new_dataset = []
    idx = 0

    paths = glob.glob(os.path.join(file_path, '*.json'))
    
    for path in paths:
        with open(path, "r") as f:
            json_data = json.load(f)

        context_info = json_data['dataset']['context_info']
        for context_data in context_info:
            context = context_data['context']
            qas = context_data['qas']

            for _qas in qas:
                question = _qas['question-1']
                answer = _qas['answer']
                question_level = _qas['question_level']
                if question_level != '상': continue
                data_row = {
                    'id': f"{file_name}_{idx}",
                    'task': 'contextqa',
                    'context': context,
                    'question': question,
                    'answer': answer,
                }
                new_dataset.append(data_row)
                idx += 1
        
        print(f"file_name:{file_name} idx:{idx}", '\t\t\t\t\t\t', end='\r')
    dataset_dict[file_name] = new_dataset
for key in dataset_dict.keys():
    print(key, len(dataset_dict[key]))
train_dataset_list = dataset_dict['VL_EE_train'] + dataset_dict['VL_NA_train'] + dataset_dict['VL_LA_train']+ dataset_dict['VL_ED_train']
eval_dataset_list = dataset_dict['VL_EE_val'] + dataset_dict['VL_NA_val'] + dataset_dict['VL_LA_val']+ dataset_dict['VL_ED_val']
with open("/workspaces/data/llm_datasets/aihub_contextqa_data_hard/기술과학/train.json", "w") as json_file:
    json.dump(train_dataset_list, json_file)
with open("/workspaces/data/llm_datasets/aihub_contextqa_data_hard/기술과학/test.json", "w") as json_file:
    json.dump(eval_dataset_list, json_file)
train_dataset_list_0 = train_dataset_list[:120000]
train_dataset_list_1 = train_dataset_list[120000:240000]
train_dataset_list_2 = train_dataset_list[240000:]

with open("/workspaces/data/llm_datasets/aihub_contextqa_data/기술과학/train_split0.json", "w") as json_file:
    json.dump(train_dataset_list_0, json_file)
    
with open("/workspaces/data/llm_datasets/aihub_contextqa_data/기술과학/train_split1.json", "w") as json_file:
    json.dump(train_dataset_list_1, json_file)
    
with open("/workspaces/data/llm_datasets/aihub_contextqa_data/기술과학/train_split2.json", "w") as json_file:
    json.dump(train_dataset_list_2, json_file)
dataset = load_dataset("/workspaces/data/llm_datasets/aihub_contextqa_data_hard/기술과학")
dataset
dataset['train'][0]
dataset = load_dataset("/workspaces/data/llm_datasets/gpt4_evol_1.3k/data/")
data = dataset['train'][0]
# answer = data['answer']
# question = data['question']
data
new_dataset = []
idx = 0
for data in dataset['train']:
    answer = data['answer']
    question = data['question']

    data_row = {
        'id': f"gpt_evol_1.3k_{idx}",
        'conversations': [
                            {'from': 'human', 'value': question},
                            {'from': 'gpt', 'value': answer},
                         ],
    }
    new_dataset.append(data_row)
    idx += 1
with open("/data/llm_datasets/custom/vicuna_format/gpt_evol_1.3k-vicuna.json", "w") as json_file:
    json.dump(new_dataset, json_file)
dataset = load_dataset("json", data_files="/data/llm_datasets/WizardLM_Orca/wizardlm_orca.json")
new_dataset = []
idx = 0
for data in dataset['train']:
    output = data['output']
    system = data['system']
    instruction = data['instruction']
    data_row = {
        'id': f"WizardLM_Orca_{idx}",
        'conversations': [
                            {'from': 'human', 'value': instruction},
                            {'from': 'gpt', 'value': output},
                         ],
        'task': 'system_instruct',
        'system': system,
    }
    new_dataset.append(data_row)
    idx += 1
    
with open("/data/llm_datasets/custom/vicuna_format/wizardlm_orca_vicuna.json", "w") as json_file:
    json.dump(new_dataset, json_file)
dataset = load_dataset("/data/llm_datasets/KoreaSpellingCorrection/")
new_dataset = []
idx = 0
for data in dataset['test']:
    wrong = data['wrong']
    correct = data['correct']
    data_row = {
        'id': f"KoreaSpelling_Correction_{idx}",
        'conversations': [
                            {'from': 'human', 'value': wrong},
                            {'from': 'gpt', 'value': correct},
                         ],
        'task': 'correction',
    }
    new_dataset.append(data_row)
    idx += 1
with open("/data/llm_datasets/custom/vicuna_format/KoreaSpellingCorrection/test.json", "w") as json_file:
    json.dump(new_dataset, json_file)
    
"""
dpo raw data load and change format
"""
dataset = load_dpo_dataset("/data/llm_datasets/truthy-dpo-v0.1/")
dataset
new_dataset = []
for data in dataset:
    new_dataset.append({
        'id': f"Ko_ultrafeedback_binarized_{data['prompt_id']}",
        'input': data['prompt'],
        'chosen': data['chosen'][1]['content'],
        'rejected': data['rejected'][1]['content'],
        'task': "dpo"
    })
new_dataset = []
# idx = 0
for idx, data in enumerate(dataset):
    new_dataset.append({
        'id': f"Ko_orca_dpo_pairs_{idx}",
        'input': data['question'],
        'chosen': data['chosen'],
        'rejected': data['rejected'],
        'task': "dpo_system",
        'system': data['system'],
    })

new_dataset = []
# idx = 0
for idx, data in enumerate(dataset):
    new_dataset.append({
        'id': f"Ko_distilabel-math-preference-dpo_{idx}",
        'input': data['instruction'],
        'chosen': data['chosen_response'],
        'rejected': data['rejected_response'],
        'task': "dpo",
    })
new_dataset = []
# idx = 0
for idx, data in enumerate(dataset):
    new_dataset.append({
        'id': f"truthy_dpo_{data['id']}",
        'input': data['prompt'],
        'chosen': data['chosen'],
        'rejected': data['rejected'],
        'system': data['system'],
        'task': "dpo_system",
    })
with open("/data/llm_datasets/custom/kodpo/untranslated/truthy-dpo-v0.1.json", "w") as json_file:
    json.dump(new_dataset, json_file)
    

class ados_DPODataset:
    def __init__(
        self, 
        dataset_path="/data/llm_datasets/custom/ados/dpo/ados_dpo_v2.json",
        eval_dataset_path = "", #/data/llm_datasets/Ultrafeedback_binarized.ko.hankang/test_prefs.json.kr
        data_format='chat-orca',
        # search_term='\n\n### Assistant:',
        num_train=None,
        num_eval=None,
    ):
        self.dataset_path = dataset_path
        self.eval_dataset_path = dataset_path
        if eval_dataset_path:
            self.eval_dataset_path = eval_dataset_path
        self.data_format = data_format
        # self.search_term = search_term
        self.num_train = num_train
        self.num_eval = num_eval
    
    def get_prompt_and_response(self, data):
        conv = get_conversation_template(self.data_format)
        if data['system']:
            conv.system_message = conv.tasks['system_instruct'].format(system=data['system'])
        conv.append_message(conv.roles[0], data['input'])
        conv.append_message(conv.roles[1], '')
        prompt = conv.get_prompt()
        conv.update_last_message(data['chosen'])
        chosen = conv.get_prompt()[len(prompt):]
        conv.update_last_message(data['rejected'])
        rejected = conv.get_prompt()[len(prompt):]
        
        return prompt, chosen, rejected
    
    def make_dpo_data_module(self):
        def split_prompt_and_responses(data) -> Dict[str, str]:
            prompt, chosen, rejected = self.get_prompt_and_response(data)
            # prompt = extract_anthropic_prompt(prompt_and_response, self.search_term)
            # promopt_rejected = extract_anthropic_prompt(prompt_and_response_rejected, self.search_term)
            return {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
                             
        train_dataset = load_dpo_dataset(self.dataset_path)
        eval_dataset = load_dpo_dataset(self.eval_dataset_path)

        original_columns = list(train_dataset.features.keys())
        original_columns_eval = list(eval_dataset.features.keys())

        if self.num_train is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), self.num_train)))
        if self.num_eval is not None:
            eval_dataset = eval_dataset.select(range(min(len(train_dataset), self.num_eval)))

        train_dataset = train_dataset.map(split_prompt_and_responses, remove_columns=original_columns)
        eval_dataset = eval_dataset.map(split_prompt_and_responses, remove_columns=original_columns_eval)

        return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)
    
""" find odd code blocks"""
dataset_path = "/data/llm_datasets/custom/ados_sft_v4.json"
# dataset = load_dataset("json", dataset_path)
dataset = load_sft_dataset(dataset_path, split=None)
train_dataset = dataset['train']

# new_dataset = []
code_prefixes = []

odd_dataset = []
oddd_dataset = []
odd_idxs = set()
normal_dataset = []
flag_normal = True
flag_code = False
for idx, data in enumerate(train_dataset):
    conversations = data['conversations']
    flag_normal = True
    for conv in conversations:
        _from = conv['from']
        if _from == 'human': continue
        _value = conv['value']
        flag_code = False
        find_iter = re.finditer('```', _value)
        temp_num = 0
        for fidx, ftext in enumerate(find_iter):
            flag_code = True
            start_index = ftext.start() + 3
            # new_dataset.append(data)
            #TODO: 스페이스바가 바로 오는 경우..
            candidate = re.split(r'[\n]', _value[start_index:])[0]
            if fidx % 2 == 0 and '```' not in candidate and candidate not in available_code_prefixes:
                odd_dataset.append((candidate, data))
                code_prefixes.append(candidate)
                odd_idxs.add(idx)
                flag_normal = False
                break
            temp_num += 1
        if temp_num % 2 != 0:
            oddd_dataset.append(('odd', data))
            odd_idxs.add(idx)
            flag_normal = False
        
        if not flag_normal:
            break
    
    if flag_code and flag_normal:
        normal_dataset.append(data)
        num_normal += 1

print(len(train_dataset), len(normal_dataset), len(odd_dataset), len(oddd_dataset), len(odd_idxs))

selected_idxs = list(range(len(train_dataset)))
for od in odd_idxs:
    selected_idxs.remove(od)
train_dataset = train_dataset.select(selected_idxs)
code_prefixes = set(code_prefixes)
code_prefixes
available_code_prefixes = set([
    '',
    'CSS',
    'HTML',
    'JavaScript',
    'Python',
    'SQL',
    'bash',
    'c',
    'c++',
    'cpp',
    'csharp',
    'css',
    'for',
    'html',
    'java',
    'javascript',
    'js',
    'json',
    'php',
    'python',
    'ruby',
    'sass',
    'scss',
    'sql',
    'sum',
    'svg',
    'swift',
    'xml',
    'yaml',
    'C#',
    'C++',
    'CSS',
    'Go',
    'HTML',
    'Java',
    'LaTeX',
    'MATLAB',
    'Markdown',
    'Proposals',
    'Python',
    'R',
    'SELECT',
    'SQL',
    'Swift',
    'VBA',
    'echo',
    'excel-vba',
    'find',
    'go',
    'gpg',
    'jsx',
    'kotlin',
    'latex',
    'markdown',
    'math',
    'matlab',
    'meditation',
    'mermaid',
    'mutt',
    'nano',
    'r',
    'rust',
    'scala',
    'sh',
    'shell',
    'sudo',
    'xpath',
    '{r}',
    '.',
    '$',
    'curl',
    'xslt',
    'Apex',
    'DAX',
    'Dockerfile',
    'apex',
    'applescript',
    'arduino',
    'asm',
    'assembly',
    'astro',
    'autoit',
    'batch',
    'bicep',
    'blade',
    'cmake',
    'cmd',
    'coffee',
    'coffeescript',
    'cql',
    'csv',
    'cypher',
    'dart',
    'delphi',
    'diff',
    'dockerfile',
    'dot',
    'emacs',
    'erb',
    'fsharp',
    'glsl',
    'gradle',
    'graphql',
    'graphviz',
    'groovy',
    'haskell',
    'hcl',
    'hlsl',
    'html+erb',
    'ini',
    'jinja',
    'ladder',
    'lasso',
    'less',
    'lisp',
    'lldb',
    'llvm',
    'logo',
    'lua',
    'makefile',
    'mathematica',
    'metal',
    'nginx',
    'nix',
    'objc',
    'objective',
    'objectivec',
    'pascal',
    'perl',
    'plaintext',
    'plantuml',
    'powershell',
    'prisma',
    'properties',
    'proto',
    'protobuf',
    'py',
    'reg',
    'rego',
    'scheme',
    'scratch',
    'solidity',
    'spss',
    'stata',
    'stencil',
    'terraform',
    'toml',
    'ts',
    'tsx',
    'txt',
    'typescript',
    'vb',
    'vba',
    'vbnet',
    'verilog',
    'yml',
    'jsp',
    'prolog',
    'razor',
    'CMD',
    'G',
    'GraphQL',
    'Makefile',
    'apache',
    'c#',
    'cython',
    'elixir',
    'jinja2',
    'julia',
    'ocaml',
    'systemverilog',
    'vbscript',
    'vhdl',
    'vue',
    'wasm',
    'wolfram',
    'zsh',
    'regex',
    ' Java',
    ' Python',
    ' c++',
    ' python',
    ' java',
    ' css',
    ' html',
    ' js',
    ' python',
    ' scala',
    'md',
    ' Bash',
    ' MATLAB',
    ' Matlab',
    ' PHP',
    ' R',
    ' Swift',
    ' bash',
    ' batteryruntime',
    ' cpp',
    ' csharp',
    ' makefile',
    ' r',
    ' ruby',
    ' sql',
    ' swift',
    ' ts',
    'ABAP',
    'AutoIt',
    'Bash',
    'C',
    'CSharp',
    'EL',
    'GDScript',
    'Kotlin',
    'M',
    'Maple',
    'MatLab',
    'Matlab',
    'PowerShell',
    'PyTorch',
    'Shell',
    'VHDL',
    'Vagrantfile',
    'XML',
    'ada',
    'afrikaans',
    'amsmath',
    'arc',
    'aspx',
    'autohotkey',
    'azurepowershell',
    'bash ',
    'bat',
    'c ',
    'cfscript',
    'cl',
    'clojure',
    'cobol',
    'command',
    'cplusplus',
    'cron',
    'cs',
    'csh',
    'dax',
    'debug',
    'deluge',
    'dos',
    'ejs',
    'excel',
    'fortran',
    'gdscript',
    'glslx',
    'golang',
    'guj',
    'handlebars',
    'js ',
    'language',
    'libraries',
    'lustrum',
    'max',
    'mikrotik',
    'npm',
    'objective-c',
    'octave',
    'ogdl',
    'pddl',
    'pine',
    'pseudo',
    'psuedocode',
    'py ',
    'qlik',
    'rb',
    'registry',
    'rpgle',
    'sas',
    'scala ',
    'scribble',
    'simple_java',
    'smalltalk',
    'soap',
    'spreadsheet',
    'tcl',
    'tex',
    'text',
    'typescript ',
    'views',
    'webidl',
    'xaml',
    'xquery',
])

# for pre in available_code_prefixes:
with open("available_code_prefixes.txt", "w") as f:
    f.write('\n'.join(available_code_prefixes))
    
%%time
"""sft translate"""
api_server_url = "http://localhost:41002"
def send_translate_request(new_dataset):
    global idx
    # for _ in range(2):
    while(1):
        if idx >= len_dataset:
            break
        lock.acquire()
        pidx = idx
        data = lang_dict['__label__en'][pidx]
        idx += 1
        lock.release()
        
        print(f"{idx}/{len_dataset}", '\t\t\t\t\t\t', end='\r')#
        
        conv = data['conversations']
        new_conv = []
        for _data in conv:
            value = _data['value']
            # response
            results = ""
            text_blocks = []
            code_blocks = []
            for bidx, block in enumerate(value.split("```")):
                if bidx % 2 == 0:
                    text_blocks.append(block)
                else:
                    code_blocks.append(block)

            for tidx, text_block in enumerate(text_blocks):
                prompt = f"### 영어:\n{text_block}\n### 한국어:\n"
                input_json = {
                    "model_name": "Gugugo-koen-7B-V1.1",
                    "prompt": prompt,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "max_new_tokens": 4096,
                    "stop": ["</끝>", "###"],
                }

                ret = requests.post(
                    api_server_url + "/worker_generate_stream",
                    json=input_json,
                    stream=True,
                )

                for chunk in ret.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if chunk:
                        result_data = json.loads(chunk.decode())

                result = result_data['text'][len(prompt):].rstrip('\n')
                results += result
                if len(code_blocks) > tidx:
                    results += "```" + code_blocks[tidx] + "```"
            
            new_conv.append({
                'from': _data['from'],
                'value': results,
            })
        new_dataset.append({
            'conversations': new_conv,
            'id': data['id'],
        })


# code_prefixes = ["python", "c++", "minikube", "docker", "json", "java", 
#                  "php", "bash", "c#", "cpp", "css", "perl", "html", "xml", 
#                  "ruby", "sql", "ini", "apt", "socat", "tcp", "localhost",
#                  "git"]
new_dataset = []
threads = []
idx = 0
lock = threading.Lock()
len_dataset = len(lang_dict['__label__en'])
n_thread = 64 * 7


for i in range(n_thread):
    t = threading.Thread(target=send_translate_request, args=(new_dataset,)) # 
    t.start()
    # time.sleep(0.5)
    threads.append(t)
    
for t in threads:
    t.join()
    
%%time
"""dpo translate"""
api_server_url = "http://localhost:21122"
def send_translate_request(new_dataset):
    global idx
    # for _ in range(2):
    while(1):
        if idx >= len_dataset:
            break
        lock.acquire()
        pidx = idx
        data = lang_dict['__label__en'][pidx]
        idx += 1
        lock.release()
        
        print(f"{idx}/{len_dataset}", '\t\t\t\t\t\t', end='\r')#
        
        system = data['system']#option
        _input = data['input']
        chosen = data['chosen']
        rejected = data['rejected']
        
        new_conv = {'system': '', 'input': '', 'chosen': '', 'rejected': ''}
        for vidx, value in enumerate([_input, chosen, rejected, system]):
            if vidx == 3:
                fidx = value.find('\n')
                if fidx != -1:
                    prefix = value[:fidx + 1]
                    value = value[fidx + 1:]
                else:
                    new_conv['system'] = value
                    continue
            
            # response
            results = ""
            text_blocks = []
            code_blocks = []
            for bidx, block in enumerate(value.split("```")):
                if bidx % 2 == 0:
                    text_blocks.append(block)
                else:
                    code_blocks.append(block)

            for tidx, text_block in enumerate(text_blocks):
                prompt = f"### 영어:\n{text_block}\n### 한국어:\n"
                input_json = {
                    "model_name": "Gugugo-koen-7B-V1.1",
                    "prompt": prompt,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "max_new_tokens": 4096,
                    "stop": ["</끝>", "###"],
                }

                ret = requests.post(
                    api_server_url + "/worker_generate_stream",
                    json=input_json,
                    stream=True,
                )

                for chunk in ret.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if chunk:
                        result_data = json.loads(chunk.decode())

                result = result_data['text'][len(prompt):].rstrip('\n')
                results += result
                if len(code_blocks) > tidx:
                    results += "```" + code_blocks[tidx] + "```"
                    
            if vidx == 0:
                new_conv['input'] = results
            elif vidx == 1:
                new_conv['chosen'] = results
            elif vidx == 2:
                new_conv['rejected'] = results
            elif vidx == 3:
                new_conv['system'] = prefix + results
            
        new_dataset.append({
            **data,
            **new_conv,
        })


new_dataset = []
threads = []
idx = 0
lock = threading.Lock()
len_dataset = len(lang_dict['__label__en'])
n_thread = 64 * 1

for i in range(n_thread):
    t = threading.Thread(target=send_translate_request, args=(new_dataset,)) # 
    t.start()
    # time.sleep(0.5)
    threads.append(t)
    
# for t in threads:
    # t.join()
    
%%time
api_server_url = "http://localhost:21122"
def send_translate_request(new_dataset):
    global idx
    # for _ in range(2):
    while(1):
        if idx >= len_dataset:
            break
        lock.acquire()
        pidx = idx
        data = dataset[pidx]
        idx += 1
        lock.release()
        
        print(f"{idx}/{len_dataset}", '\t\t\t\t\t\t', end='\r')#
        
        text = tokenizer.decode(data['input_ids'])
        sentences = text.split('.')

        len_split = 1024
        splits = []
        split = ""
        for sentence in sentences:
            if len(split) < len_split:
                split += sentence + "."
            else:
                splits.append(split)
                split = ""
        
        result_sentences = []
        for value in splits:
            # response
            results = ""
            text_blocks = []
            code_blocks = []
            for bidx, block in enumerate(value.split("```")):
                if bidx % 2 == 0:
                    text_blocks.append(block)
                else:
                    code_blocks.append(block)

            for tidx, text_block in enumerate(text_blocks):
                prompt = f"### 영어:\n{text_block}\n### 한국어:\n"
                input_json = {
                    "model_name": "Gugugo-koen-7B-V1.1",
                    "prompt": prompt,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "max_new_tokens": 4096,
                    "stop": ["</끝>", "###"],
                }

                ret = requests.post(
                    api_server_url + "/worker_generate_stream",
                    json=input_json,
                    stream=True,
                )

                for chunk in ret.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if chunk:
                        result_data = json.loads(chunk.decode())

                result = result_data['text'][len(prompt):].rstrip('\n')
                results += result
                if len(code_blocks) > tidx:
                    results += "```" + code_blocks[tidx] + "```"
                    
            result_sentences.append(results)
            
        new_dataset.append(result_sentences)


new_dataset = []
threads = []
idx = 0
lock = threading.Lock()
len_dataset = len(dataset)
n_thread = 64 * 1

for i in range(n_thread):
    t = threading.Thread(target=send_translate_request, args=(new_dataset,)) # 
    t.start()
    # time.sleep(0.5)
    threads.append(t)
    
for t in threads:
    t.join()
    

# 이후 dedup
# def dedup_by_similarity(dataset, prompt_template='chat-orca', target_text_len=100, n_results=100, distance_threshold = 0.6):
# dataset
prompt_template='dpo'
target_text_len=100
n_results=100
distance_threshold = 0.35
    
if prompt_template == 'chat-orca':
    conv = get_conversation_template(prompt_template)
    system_message = conv.system_message
    sep_style = conv.sep_style
    sep = conv.sep
    prompt_user, prompt_bot = conv.roles

    len_sep_style = 0
    if sep_style == SeparatorStyle.ADD_COLON_TWO:
        len_sep_style = 1

    len_front = len(system_message) + len(sep) + len(prompt_user) + len_sep_style + 1
    len_rear = len(sep) + len(prompt_bot) + len_sep_style
    def filter_question(data):
        return { 
            # **data,
            'prompt': data['prompt'][len_front:-len_rear][:target_text_len]
        }

if prompt_template == 'vicuna':
    def filter_question(data):
        return {
            'prompt': data['conversations'][0]['value'][:target_text_len]
        }

if prompt_template == 'dpo':
    def filter_question(data):
        return {
            'prompt': data['input'][:target_text_len]
        }
    
question_dataset = dataset.map(filter_question)

chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
embedder = get_embedder("ddobokki/klue-roberta-base-nli-sts-ko-en")
collection = chroma_client.create_collection(name="context", embedding_function=embedder.embed, metadata={"hnsw:space": "cosine"})
ids = []
# add
texts = question_dataset['prompt']
last_id = -1
new_ids = [f"id{i+last_id+1}" for i in range(len(texts))]
ids += new_ids
collection.add(documents=texts, ids=new_ids)

query_ids = copy.deepcopy(new_ids)
selected_ids = []
duplicated_ids = []

weird_ids = []
error_ids = []
while query_ids:
    current_id = random.choice(query_ids)
    if current_id in selected_ids:
        print("Warning: this is weird..")
        weird_ids.append(current_id)
        continue
    selected_ids.append(current_id)
    search_strings = [texts[int(current_id[2:])]]
    if collection.count() == 0:
        print("Warning: collection is empty. Forced break")
        break
    result = collection.query(query_texts=search_strings, n_results=min(n_results, len(query_ids)), include=['distances']) #'documents'

    search_ids = result['ids'][0]
    distances = result['distances'][0]
    # documents = result['documents'][0]
    remove_ids = []
    for idx in range(len(search_ids)):
        sid = search_ids[idx]
        dist = distances[idx]
        if dist < distance_threshold:
            remove_ids.append(sid)

    for rid in remove_ids:
        if rid in query_ids:
            query_ids.remove(rid)

    if remove_ids:
        duplicated_ids += remove_ids
        collection.delete(ids=remove_ids)
    else:
        print("Warning: this is error..")
        error_ids.append(current_id)

    print(f"Total:{len(new_ids)} Selected:{len(selected_ids)} current_dup:{len(remove_ids)} vector_store:{collection.count()} remained:{len(query_ids)} total_dup:{len(duplicated_ids)}", '\t\t\t\t\t', end='\r')

print('finished dedup data:', f"Total:{len(new_ids)} Selected:{len(selected_ids)} current_dup:{len(remove_ids)} vector_store:{collection.count()} remained:{len(query_ids)} total_dup:{len(duplicated_ids)}")

selected_ids = [int(sid[2:]) for sid in set(selected_ids)]

dataset = dataset.select(selected_ids)

# return dataset
def dedup_non_pair(dataset, data_format='sft'):
    def validate_non_pair(data):
        dedup_flag = False
        if data_format == 'sft':
            conversations = data['conversations']

            if len(conversations) == 0: # empty
                return False

            if conversations[0]["from"] != 'human': # skip first if it's not human
                conversations = conversations[1:]

            for idx, conv in enumerate(conversations): # check right pairs
                role = conv['from']
                if idx % 2 == 0 and role != 'human':
                    dedup_flag = True
                    break
                elif idx % 2 == 1 and role != 'gpt':
                    dedup_flag = True
                    break
        elif data_format == 'dpo':
            if data['input'] and data['chosen'] and data['rejected']:
                return True
            else:
                return False
                
        # if dedup_flag:
        #     print(conv)
        
        return not dedup_flag
    
    start = len(dataset)
    dataset = dataset.filter(validate_non_pair)
    print(f"{start - len(dataset)}/{start} deduped")
    return dataset
def dedup_too_much_token(dataset, data_format='sft', max_token=3800):
    api_server_url = "http://localhost:21122"
    def validate_too_much_token(data):
        dedup_flag = False
        if data_format == 'sft':
            convs = data['conversations']
        elif data_format == 'dpo':
            convs = [data['input'], data['chosen'], data['rejected']]
        else:
            print("data_format should be [sft, dpo]")
            return
        
        num_token = 0
        for conv in convs:
            if data_format == 'sft':
                _from = conv['from']
                _value = conv['value']
            else:
                _value = conv
            
            input_json = {
                "model_name": "DIE_10.7b_sft_v4_dpo_v2_ep3",
                "prompt": _value,
            }

            ret = requests.post(api_server_url + "/count_token", json=input_json)

            output_json = ret.json()
            num_token += output_json['count']
            
            if num_token > max_token:
                dedup_flag = True
                break

        # if dedup_flag:
        #     print(conv)
        
        return not dedup_flag
    
    start = len(dataset)
    dataset = dataset.filter(validate_too_much_token)
    print(f"{start - len(dataset)}/{start} deduped")
    return dataset

""" find odd code blocks(dpo)"""
dataset_path = dpo_list4[2]
print(dataset_path)
# dataset = load_dataset("json", dataset_path)
dataset = load_dpo_dataset(dataset_path, split=None)
train_dataset = dataset['train']

# new_dataset = []
code_prefixes = []

odd_dataset = []
oddd_dataset = []
odd_idxs = set()
normal_dataset = []
flag_normal = True
flag_code = False
num_normal = 0
for idx, data in enumerate(train_dataset):
    # conversations = data['conversations']
    flag_normal = True
    for conv in [data['input'], data['chosen'], data['rejected']]:
        _value = conv
        flag_code = False
        find_iter = re.finditer('```', _value)
        temp_num = 0
        for fidx, ftext in enumerate(find_iter):
            flag_code = True
            start_index = ftext.start() + 3
            # new_dataset.append(data)
            #TODO: 스페이스바가 바로 오는 경우..
            candidate = re.split(r'[\n]', _value[start_index:])[0]
            if fidx % 2 == 0 and '```' not in candidate and candidate not in available_code_prefixes:
                odd_dataset.append((candidate, data))
                code_prefixes.append(candidate)
                odd_idxs.add(idx)
                flag_normal = False
                break
            temp_num += 1
        if temp_num % 2 != 0:
            oddd_dataset.append(('odd', data))
            odd_idxs.add(idx)
            flag_normal = False
        
        if not flag_normal:
            break
    
    if flag_code and flag_normal:
        normal_dataset.append(data)
        num_normal += 1

print(len(train_dataset), len(normal_dataset), len(odd_dataset), len(oddd_dataset), len(odd_idxs))

new_dataset = []
for data in dataset_:
    new_dataset.append(data)
    
with open('/data/llm_datasets/custom/kodpo/deduped/ko_ultrafeedback_binarized.json', "w") as json_file:
    json.dump(new_dataset, json_file)
    
from transformers import AutoTokenizer
# model_path = "/workspaces/disk0/data/llm_weights/Platypus2-70B-instruct/"
# model_path = "/workspaces/disk0/data/llm_weights/vicuna-13b-v1.5/"
model_path = "/workspaces/disk0/data/llm_weights/MoMo-70B-lora-1.8.4-DPO/"
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    model_max_length=4096,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.unk_token
from transformers.trainer_pt_utils import LabelSmoother
import torch
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
raw_data = load_sft_dataset("/data/llm_datasets/custom/ados/sft/ados_sft_v4.1.json")
from fastchat.train.train import LazySupervisedDataset
dataset = LazySupervisedDataset(raw_data, tokenizer, data_format="vicuna")

sources = [raw_data[50000]["conversations"]]
data_format="qwen"
custom_system_message="hello."

conv = get_conversation_template(data_format)
roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

if custom_system_message is not None:
    conv.system_message = custom_system_message

# Apply prompt templates for qwen
if data_format == "qwen":
    assert tokenizer("<|im_end|>").input_ids[0] == 151645

    max_len = tokenizer.model_max_length
    im_start = 151644
    im_end = 151645
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(conv.system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)
    
from transformers.trainer_pt_utils import LabelSmoother
import torch
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
tokenizer.decode(torch.where(targets[0] == IGNORE_TOKEN_ID, tokenizer.unk_token_id, targets[0]))