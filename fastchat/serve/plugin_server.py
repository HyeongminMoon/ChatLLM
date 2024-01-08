import argparse
import requests
import uvicorn
import time
import json
import tempfile
from typing import List
from datetime import datetime
import threading
import os

import docx2txt
from pypdf import PdfReader
from langdetect import detect

from fastapi import FastAPI, Request, BackgroundTasks, Form, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from fastchat.modules.keyword_generator import generate_keyword
from fastchat.modules.vector_store import ChromaCollector, get_websearch_result, feed_data_into_collector
from fastchat.modules.translator import translate, translators_pool, get_languages
from fastchat.model.model_adapter import get_conversation_template
from fastchat.utils import build_logger
from vllm.utils import random_uuid

from fastchat.modules.constant_plugin import (
    LOGDIR,
    SESSION_CONTROLLER_HEARTBEAT_TIME,
    PluginErrorCode,
)

from fastchat.conversation import SeparatorStyle

worker_controller_url = "http://localhost:21001"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger = build_logger("api_server", f"api_server.log")

class SessionController:
    def __init__(self):
        self.vector_stores = {}
        self.memories = {}
        self.heart_beat_thread = threading.Thread(
            target=heart_beating_session_controller, 
            args=[self,],
            daemon=True,
        )
        self.heart_beat_thread.start()
    
    def check_timeout(self):
        # vector store 삭제
        removal_ids = []
        for session_id in self.vector_stores:
            if self.vector_stores[session_id]["timeout"] < time.time():
                removal_ids.append(session_id)
            else:
                self.vector_stores[session_id]["lifetime"] += SESSION_CONTROLLER_HEARTBEAT_TIME
        for session_id in removal_ids:
            self.remove_vector_store(session_id)
        
        # memory 삭제
        removal_ids = []
        for session_id in self.memories:
            if self.memories[session_id]["timeout"] < time.time():
                removal_ids.append(session_id)
        for session_id in removal_ids:
            self.remove_memory(session_id)
            
        logger.info(f"#Active({len(self.memories.keys())}), VS:{self.vector_stores}")
    
    def add_vector_store(self, session_id, lifespan=3600):
        if session_id not in self.vector_stores:
            self.vector_stores[session_id] = {
                "vector_store": ChromaCollector(),
                "lifespan": lifespan,
                "lifetime": 0,
                "timeout": int(time.time()) + lifespan
            }
            return PluginErrorCode.NULL
        else:
            return PluginErrorCode.VECTOR_STORE_ALREADY_EXIST # already exists
        
    def remove_vector_store(self, session_id):
        if session_id in self.vector_stores:
            del self.vector_stores[session_id]
            return PluginErrorCode.NULL
        else:
            return PluginErrorCode.VECTOR_STORE_NOT_EXIST # not exists
    
    def clear_vector_store(self, session_id):
        if session_id in self.vector_stores:
            self.vector_stores[session_id]["vector_store"].clear()
            return PluginErrorCode.NULL
        else:
            return PluginErrorCode.VECTOR_STORE_NOT_EXIST # not exists
        
    def get_vector_store(self, session_id):
        if session_id in self.vector_stores:
            return self.vector_stores[session_id]["vector_store"]
        return PluginErrorCode.VECTOR_STORE_NOT_EXIST
    
    def add_memory(self, session_id, model_name, lifespan=3600):
        if session_id not in self.memories:
            self.memories[session_id] = {
                "conv": get_conversation_template(model_name),
                "context": "",
                "last_prompt": "",
                "timeout": int(time.time()) + lifespan,
                # "urls": "",
            }
            return PluginErrorCode.NULL
        else:
            return PluginErrorCode.SESSION_MEMORY_ALREADY_EXIST
    
    def remove_memory(self, session_id):
        if session_id in self.memories:
            del self.memories[session_id]
            return PluginErrorCode.NULL
        else:
            return PluginErrorCode.SESSION_MEMORY_NOT_EXIST
    
    def clear_memory(self, session_id):
        if session_id in self.memories:
            template_name = self.memories[session_id]["conv"].name
            timeout = self.memories[session_id]["timeout"]
            self.memories[session_id] = {
                "conv": get_conversation_template(template_name),
                "context": "",
                "last_prompt": "",
                "timeout": timeout,
            }
            return PluginErrorCode.NULL
        else:
            return PluginErrorCode.SESSION_MEMORY_NOT_EXIST # not exists
    
    def get_memory(self, session_id):
        if session_id in self.memories:
            return self.memories[session_id]
        return PluginErrorCode.SESSION_MEMORY_NOT_EXIST
    

def heart_beating_session_controller(session_controller):
    while True:
        time.sleep(SESSION_CONTROLLER_HEARTBEAT_TIME)
        session_controller.check_timeout()

# sub func of @app.post("/websearch_run_all")
def generate_websearch_result(collector, query):
    for search_contents, urls, is_yield in get_websearch_result(collector, query):
        if is_yield:
            ret = {
                "task_status": search_contents, 
                "urls": [],
                "context": "",
            }
            yield (json.dumps(ret) + "\0").encode()
    
    if not urls:
        ret = {
            "task_status": search_contents, 
            "urls": [],
            "context": "",
        }
    else:
        ret = {
            "task_status": "Finished", 
            "urls": urls,
            "context": (
                f"\nDate:{datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                "Your reply should be based on the context below.\n\nContext:\n"
                f"{search_contents}"
            ),
        }
    
    yield (json.dumps(ret) + "\0").encode()
    return

# sub func of @app.post("/worker_generate_stream")
def generate_stream_result(model_url, params, is_auto=False):
    if is_auto:
        memory = controller.get_memory(params["session_id"])
        conv = memory["conv"]
        params["stop"] = conv.stop_str
        params["stop_token_ids"]: conv.stop_token_ids
    
    ret = requests.post(
        model_url + "/worker_generate_stream",
        json=params, 
        stream=True,
    )
    
    bot_answer = ''
    for chunk in ret.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            if is_auto:
                data = json.loads(chunk.decode())
                if conv.sep_style == SeparatorStyle.ADD_COLON_SINGLE or \
                    conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
                    bot_answer = data["text"].split(f"{conv.roles[1]}:")[-1]
                elif conv.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
                    bot_answer = data["text"].split(f"{conv.roles[1]}\n")[-1]
                else:
                    bot_answer = data["text"].split(f"{conv.roles[1]}")
                    logger.warn("There's no defined split method for bot_answer")
                conv.update_last_message(bot_answer)
                yield json.dumps({"text": bot_answer}).encode() + b"\0"
            else:
                yield chunk + b"\0"
          
    
    with open(get_conv_log_filename(), "a", encoding='utf-8') as fout:
        data = {
            "time": round(time.time(), 4),
            "session_id": params.get("session_id"),
            "prompt": params['prompt'],
            "bot_answer": bot_answer,
        }

        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    return

def valid_input_checker(params, values: List[str]):
    not_exists = []
    for value in values:
        if value not in params:
            not_exists.append(value)
    return not_exists

def get_conv_log_filename():
    t = datetime.now()
    os.makedirs(LOGDIR, exist_ok=True)
    name = os.path.join(LOGDIR, f"OLLM-{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name

def background_save_task_auto(session_id, params):
    async def save_conv() -> None:
        with open(get_conv_log_filename(), "a", encoding='utf-8') as fout:
            memory = controller.get_memory(session_id)
            conv = memory["conv"]
            data = {
                "time": round(time.time(), 4),
                "session_id": session_id,
                "prompt": params['prompt'],
                "bot_answer": conv.messages[-1][-1],
            }

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    background_tasks = BackgroundTasks()
    background_tasks.add_task(save_conv)
    return background_tasks

@app.get("/list_models")
async def api_list_models(request: Request):
    ret = requests.post(
        worker_controller_url + "/list_models"
    )
    return ret.json()

@app.get("/refresh_all_models")
async def api_refresh_models(request: Request):
    ret = requests.post(
        worker_controller_url + "/refresh_all_workers"
    )

@app.get("/get_random_uuid")
async def api_get_random_uuid(request: Request):
    return {"uuid": random_uuid()}


@app.post("/model_details")
async def api_model_details(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["model_name"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    
    model_name = params["model_name"]
    ret = requests.post(
        worker_controller_url + "/get_worker_address", json={"model": model_name}
    )
    model_url = ret.json()["address"]
    ret = requests.post(
        model_url + "/model_details"
    )
    return ret.json()

@app.post("/worker_get_conv_template")
async def api_worker_get_conv_template(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["model_name"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    model_name = params["model_name"]
    ret = requests.post(
        worker_controller_url + "/get_worker_address", json={"model": model_name}
    )
    model_url = ret.json()["address"]
    ret = requests.post(
        model_url + "/worker_get_conv_template"
    )
    return ret.json()

@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["model_name", "prompt"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    model_name = params["model_name"]
    
    ret = requests.post(
        worker_controller_url + "/get_worker_address", json={"model": model_name}
    )
    model_url = ret.json()["address"]
    ret = requests.post(
        model_url + "/count_token", json=params
    )
    return ret.json()

@app.post("/worker_generate")
async def api_worker_generate(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["model_name", "prompt"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    model_name = params["model_name"]
    ret = requests.post(
        worker_controller_url + "/get_worker_address", json={"model": model_name}
    )
    model_url = ret.json()["address"]
    ret = requests.post(
        model_url + "/worker_generate", json=params
    )
    return ret.json()
  
@app.post("/worker_generate_stream")
async def api_worker_gernate_stream(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["model_name", "prompt"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    model_name = params["model_name"]
    ret = requests.post(
        worker_controller_url + "/get_worker_address", json={"model": model_name}
    )
    model_url = ret.json()["address"]
    generator = generate_stream_result(model_url, params)
    return StreamingResponse(generator)

@app.post("/worker_generate_auto")
async def api_worker_generate_auto(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["model_name", "user_input"]) #"session_id"
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    model_name = params["model_name"]
    user_input = params["user_input"]
    # session_id = params["session_id"]
    session_id = params.get("session_id")
    context = params.get("context")
    
    if session_id is None:
        session_id = random_uuid()
    
    ret = requests.post(
        worker_controller_url + "/get_worker_address", json={"model": model_name}
    )
    model_url = ret.json()["address"]
    
    memory = controller.get_memory(session_id)
    if memory == PluginErrorCode.SESSION_MEMORY_NOT_EXIST:
        controller.add_memory(session_id, model_name)
        memory = controller.get_memory(session_id)
    
    conv =  memory["conv"]
    conv.append_message(conv.roles[0], user_input)
    conv.append_message(conv.roles[1], None)
    if context is None and memory["context"]:
        context = memory["context"]
    prompt = conv.get_prompt(context=context)
    memory["last_prompt"] = prompt
    
    params["prompt"] = prompt
    # count_token
    ret = requests.post(
        model_url + "/model_details"
    )
    max_token_len = ret.json()["context_length"]
    ret = requests.post(
        model_url + "/count_token", json=params
    )
    num_token = ret.json()["count"]
    if num_token > max_token_len:
        return {"err_code": PluginErrorCode.EXCEED_CONTEXT_LENGTH}
    # websearch 사용여부?
    
    ret = requests.post(
        model_url + "/worker_generate", json=params
    )
    
    # 파싱하여 assistant 대답만 추출
    result = ret.json()
    if conv.sep_style == SeparatorStyle.ADD_COLON_SINGLE or \
        conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
        bot_answer = result["text"].split(f"{conv.roles[1]}:")[-1]
    elif conv.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
        bot_answer = result["text"].split(f"{conv.roles[1]}\n")[-1]
    else:
        bot_answer = result["text"].split(f"{conv.roles[1]}")
        logger.warn("There's no defined split method for bot_answer")
        
    conv.update_last_message(bot_answer)
    return {"text": bot_answer}

# @app.options("/worker_generate_stream_auto")
# async def options_worker_gernate_stream_auto():
#     # OPTIONS 요청에 대한 처리
#     return Response(status_code=200)

@app.post("/worker_generate_stream_auto")
async def api_worker_gernate_stream_auto(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["model_name", "user_input", "session_id"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    model_name = params["model_name"]
    user_input = params["user_input"]
    session_id = params["session_id"]
    context = params.get("context")
    
    ret = requests.post(
        worker_controller_url + "/get_worker_address", json={"model": model_name}
    )
    model_url = ret.json()["address"]
    
    memory = controller.get_memory(session_id)
    if memory == PluginErrorCode.SESSION_MEMORY_NOT_EXIST:
        controller.add_memory(session_id, model_name)
        memory = controller.get_memory(session_id)
    
    conv =  memory["conv"]
    conv.append_message(conv.roles[0], user_input)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt(context=context)
    memory["last_prompt"] = prompt
    params["prompt"] = prompt
    # count_token
    ret = requests.post(
        model_url + "/model_details"
    )
    max_token_len = ret.json()["context_length"]
    ret = requests.post(
        model_url + "/count_token", json=params
    )
    num_token = ret.json()["count"]
    if num_token > max_token_len:
        return {"err_code": PluginErrorCode.EXCEED_CONTEXT_LENGTH}
    
    # background_tasks = background_save_task_auto(session_id, params)
    
    generator = generate_stream_result(model_url, params, True)
    return StreamingResponse(generator) #, background=background_tasks

@app.post("/worker_get_last_prompt")
async def api_worker_get_last_prompt(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["session_id"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    # model_name = params["model_name"]
    session_id = params["session_id"]
    memory = controller.get_memory(session_id)
    if memory == PluginErrorCode.SESSION_MEMORY_NOT_EXIST:
        return {"prompt": ""}
    
    return {"prompt": memory["last_prompt"]}

@app.post("/session_history_clear")
async def api_session_history_clear(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["session_id"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    # model_name = params["model_name"]
    session_id = params["session_id"]
    memory = controller.clear_memory(session_id)
    if memory == PluginErrorCode.SESSION_MEMORY_NOT_EXIST:
        return {"err_code": PluginErrorCode.SESSION_MEMORY_NOT_EXIST}
    
    return {"err_code": PluginErrorCode.NULL}

@app.post("/worker_regenerate_auto")
async def api_worker_regenerate_auto(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["model_name", "session_id"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    model_name = params["model_name"]
    session_id = params["session_id"]
    
    ret = requests.post(
        worker_controller_url + "/get_worker_address", json={"model": model_name}
    )
    model_url = ret.json()["address"]
    
    memory = controller.get_memory(session_id)
    if memory == PluginErrorCode.SESSION_MEMORY_NOT_EXIST:
        return {"err_code": PluginErrorCode.SESSION_MEMORY_NOT_EXIST}
    
    prompt = memory["last_prompt"]
    if not prompt:
        return {"err_code": PluginErrorCode.EMPTY_SESSION}
    
    # conv에서 미지막 대답 지우기
    conv = memory["conv"]
    conv.update_last_message("")
    
    params["prompt"] = prompt
    ret = requests.post(
        model_url + "/worker_generate", json=params
    )
    
    # 파싱하여 assistant 대답만 추출
    result = ret.json()
    if conv.sep_style == SeparatorStyle.ADD_COLON_SINGLE or \
        conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
        bot_answer = result["text"].split(f"{conv.roles[1]}:")[-1]
    elif conv.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
        bot_answer = result["text"].split(f"{conv.roles[1]}\n")[-1]
    else:
        bot_answer = result["text"].split(f"{conv.roles[1]}")
        logger.warn("There's no defined split method for bot_answer")
    conv.update_last_message(bot_answer)
    return {"text": bot_answer}
    
@app.post("/worker_regenerate_stream_auto")
async def api_worker_regenerate_stream_auto(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["model_name", "session_id"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    model_name = params["model_name"]
    session_id = params["session_id"]
    
    ret = requests.post(
        worker_controller_url + "/get_worker_address", json={"model": model_name}
    )
    model_url = ret.json()["address"]
    
    memory = controller.get_memory(session_id)
    if memory == PluginErrorCode.SESSION_MEMORY_NOT_EXIST:
        return {"err_code": PluginErrorCode.SESSION_MEMORY_NOT_EXIST}
    
    prompt = memory["last_prompt"]
    if not prompt:
        return {"err_code": PluginErrorCode.EMPTY_SESSION}
    
    # conv에서 미지막 대답 지우기
    conv = memory["conv"]
    conv.update_last_message("")
    
    params["prompt"] = prompt
    
    # background_tasks = background_save_task_auto(session_id, params)
    # 파싱하여 assistant 대답만 추출
    generator = generate_stream_result(model_url, params, True)
    return StreamingResponse(generator) #, background=background_tasks

# @app.options("/worker_stop_stream")
# async def options_worker_stop_stream():
#     # OPTIONS 요청에 대한 처리
#     return Response(status_code=200)

@app.post("/worker_stop_stream")
async def api_worker_stop_stream(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["model_name", "session_id"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    model_name = params["model_name"]
    ret = requests.post(
        worker_controller_url + "/get_worker_address", json={"model": model_name}
    )
    model_url = ret.json()["address"]
    ret = requests.post(
        model_url + "/worker_stop_stream", json=params
    )
    return ret.json()

@app.post("/websearch_generate_keyword")
async def api_generate_keyword(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["model_name", "query", "session_id"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    model_name = params["model_name"]
    query = params["query"]
    session_id = params["session_id"]
    # params["request_id"] = session_id
    
    search_keywords = generate_keyword(
        model_name, 
        query, 
        session_id=session_id,
    )
    
    return {"search_keyword": search_keywords[0]}

@app.post("/websearch_run_all")
async def api_websearch_run_all(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["model_name", "session_id", "search_keyword"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    model_name = params["model_name"]
    session_id = params["session_id"]
    search_keyword = params["search_keyword"]
    
    vector_store = controller.get_vector_store(session_id)
    if vector_store == PluginErrorCode.VECTOR_STORE_NOT_EXIST:
        controller.add_vector_store(session_id)
        vector_store = controller.get_vector_store(session_id)
    else:
        vector_store.clear()
    
    generator = generate_websearch_result(vector_store, search_keyword)
    
    # background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator) #, background=background_tasks

@app.get("/retrievalqa_details")
async def api_retrievalqa_details(request: Request):
    return {"available_formats": ['.pdf', '.docx', '.txt']}

@app.post("/retrievalqa_input_doc")
async def api_retrievalqa_input_doc(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    vector_store = controller.get_vector_store(session_id)
    if vector_store == PluginErrorCode.VECTOR_STORE_NOT_EXIST:
        controller.add_vector_store(session_id)
        vector_store = controller.get_vector_store(session_id)
    
    file_name = file.filename

    try:
        if file_name.lower().endswith('.pdf'):
            reader = PdfReader(file.file)
            texts = []
            for page in reader.pages:
                texts.append(page.extract_text())
            text = '\n'.join(texts)
        elif file_name.lower().endswith('.docx'):
            b_data = file.file.read()
            temp_file = tempfile.NamedTemporaryFile()
            temp_name = temp_file.name
            temp_file.seek(0)
            temp_file.write(b_data)
            text = docx2txt.process(temp_name) #, "../langchain/images/"
        elif file_name.lower().endswith('.txt'):
            text = file.file.read()

        feed_data_into_collector(vector_store, text)
    except Exception as e:
        return {"err_code": PluginErrorCode.PROCESSING_DOCS_FAIL, "err_msg": str(e)}
    
    return {"err_code": PluginErrorCode.NULL}
    
    
@app.post("/retrievalqa_get_result")
async def api_retrievalqa_get_result(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["query", "session_id"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    session_id = params["session_id"]
    query = params["query"]
    
    vector_store = controller.get_vector_store(session_id)
    if vector_store == PluginErrorCode.VECTOR_STORE_NOT_EXIST:
        return {"err_code": PluginErrorCode.VECTOR_STORE_NOT_EXIST, "err_msg": "session does not exist"}
    
    result = vector_store.collection.query(query_texts=[query], n_results=5, include=['documents', 'distances'])
    
    search_contents = '\n'.join(result['documents'][0])
    context = (
        f"\nDate:{datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        "Your reply should be based on the context below.\n\nContext:\n"
        f"{search_contents}"
    )
    
    return {"context": context}
    
@app.post("/vector_store_create")
async def api_vector_store_create(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["session_id"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    session_id = params["session_id"]
    
    ret = controller.add_vector_store(session_id)
    
    if ret != PluginErrorCode.NULL:
        return {"err_code": ret}
    
    return {"err_code": PluginErrorCode.NULL}
    
@app.post("/vector_store_remove")
async def api_vector_store_remove(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["session_id"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    session_id = params["session_id"]
    
    ret = controller.remove_vector_store(session_id)
    
    if ret != PluginErrorCode.NULL:
        return {"err_code": ret}
    
    return {"err_code": PluginErrorCode.NULL}
    
@app.post("/vector_store_clear")
async def api_vector_store_clear(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["session_id"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    session_id = params["session_id"]
    
    ret = controller.clear_vector_store(session_id)
    
    if ret != PluginErrorCode.NULL:
        return {"err_code": ret}
    
    return {"err_code": PluginErrorCode.NULL}

@app.post("/vector_store_details")
async def api_vector_store_details(request: Request):
    params = await request.json()
    
    if "session_id" in params:
        session_id = params["session_id"]
        vs = controller.vector_stores.get(session_id)
        if vs is None:
            return {"err_code": PluginErrorCode.VECTOR_STORE_NOT_EXIST, "err_msg": "session does not exist"}
        
        active_stores = [{
            "session_id": session_id,
            "num_chucks": len(vs["vector_store"].ids),
            "lifespan": vs["lifespan"],
            "lifetime": vs["lifetime"],
            "timeout": vs["timeout"],
        }]
    else:
        active_stores = []
        for session_id, vs in controller.vector_stores.items():
            active_stores.append({
                "session_id": session_id,
                "num_chucks": len(vs["vector_store"].ids),
                "lifespan": vs["lifespan"],
                "lifetime": vs["lifetime"],
                "timeout": vs["timeout"],
            })
            
    return {"active_stores": active_stores} 

@app.post("/ts_translate")
async def api_ts_translate(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["sentence"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    sentence = params["sentence"]
    from_lang = params["from_lang"] if "from_lang" in params else "en"
    to_lang = params["to_lang"] if "to_lang" in params else "ko"
    ts_tool = params["ts_tool"] if "ts_tool" in params else "google"
    
    result = translate(
        sentence, 
        ts_tool=ts_tool, 
        ts_lang=to_lang, 
        from_lang=from_lang
    )
    
    return {"result": result}

@app.post("/ts_detect_language")
async def api_ts_detect_language(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["sentence"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    sentence = params["sentence"]
    
    detected_lang = detect(sentence)
    
    return {"detected_lang": detected_lang}

@app.get("/ts_pool")
async def api_ts_pool(request: Request):
    return {"ts_tools": translators_pool}

@app.post("/ts_get_languages")
async def api_ts_get_languages(request: Request):
    params = await request.json()
    not_exists = valid_input_checker(params, ["ts_tool"])
    if not_exists:
        return {"err_code": PluginErrorCode.INVALID_INPUT, "err_msg": f"invalid input: {str(not_exists)}"}
    ts_tool = params["ts_tool"]
    
    return {"available_languages": get_languages(ts_tool)}
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=41001)
    args = parser.parse_args()
    
    logger.info(f"args: {args}")
    
    controller = SessionController()
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")