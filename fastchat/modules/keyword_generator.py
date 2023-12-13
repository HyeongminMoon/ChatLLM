import requests
import json

# model_name = "MingAI-70B-chat-orca_v0.4_v0.2_retrained_checkpoint-2-GPTQ"
# query = "아도스 연봉은 얼마야?"

def generate_keyword(
    model_name,
    query,
    controller_url = "http://localhost:21001",
    temperature = 0,
    top_p = 0,
    repetition_penalty = 1,
    max_new_tokens = 256,
    stop_str = ",",
    stop_token_ids = None,
    session_id = None,
):
    # Query worker address
    ret = requests.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )

    worker_addr = ret.json()["address"]

    # No available worker
    if worker_addr == "":
        return []

    # Construct Prompt
    prompt = (f"### System:\nThis is a system prompt, please behave and help the user.\n\n"
              f"### User: I want you to act as a google search keywords generator. I will speak to you questions. "
              f"I want you to only reply with a few search sentences based on the questions. "
              f"You must distinguish each sentence using comma(,)"
              f"do not write explanations. My first command is {query}\n\n### Assistant:"
    )
    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "stop": stop_str,
        "stop_token_ids": stop_token_ids,
        "echo": False,
        "session_id": session_id,
    }

    # Stream output
    response = requests.post(
        worker_addr + "/worker_generate",
        headers={"User-Agent": "FastChat Client"},
        json=gen_params,
        # stream=True,
        timeout=100,
    )

    try:
        data = json.loads(response.text)
        search_keywords = data['text'].lstrip().split(',')

        return search_keywords
    except:
        return query