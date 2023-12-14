"""Benchmarking script to test the throughput of serving workers."""
import argparse
import json
import os

import requests
import threading
import time
import pandas as pd

from fastchat.model.model_adapter import get_conversation_template

test_input = "Tell me a story."
test_n_threads = [1, 2, 4, 16, 32, 64]
test_file_name = "throughput.csv"


def main():
    controller_addr = args.controller_address
    ret = requests.post(controller_addr + "/refresh_all_workers")

    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": args.model_name}
    )
    worker_url = ret.json()["address"]

    if worker_url == "":
        print(f"no available worker for {args.model_name}")
        return

    conv = get_conversation_template(args.model_name)
    conv.append_message(conv.roles[0], test_input)
    prompt_template = conv.get_prompt()

    def send_request(results, i):
        thread_worker_addr = worker_url
        # print(f"thread {i} goes to {thread_worker_addr}")
        response = requests.post(
            thread_worker_addr + "/worker_generate_stream",
            headers=headers,
            json=ploads[i],
            stream=False,
        )
        k = list(
            response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0")
        )
        response_new_words = json.loads(k[-2].decode("utf-8"))["text"]
        results[i] = response_new_words

    for n_thread in test_n_threads:
        prompts = [prompt_template for _ in range(n_thread)]

        headers = {"User-Agent": "fastchat Client"}
        ploads = [
            {
                "model": args.model_name,
                "prompt": prompts[i],
                "max_new_tokens": 2048,
                "temperature": 0.0,
                "top_p": 0.0,
            }
            for i in range(len(prompts))
        ]

        # use N threads to prompt the backend
        tik = time.time()
        threads = []
        results = [None] * n_thread
        for i in range(n_thread):
            t = threading.Thread(target=send_request, args=(results, i))
            t.start()
            # time.sleep(0.5)
            threads.append(t)

        for t in threads:
            t.join()

        if n_thread == 1:
            print(f"Sample Result: {results[0]}")
        print(f"Time (POST): {time.time() - tik} s")

        time_seconds = time.time() - tik

        # count tokens
        ret = requests.post(worker_url + "/count_token", json={"prompt": test_input})
        count_q = ret.json()["count"]
        total_count = 0
        for i in range(n_thread):
            result = results[i]
            ret = requests.post(worker_url + "/count_token", json={"prompt": result})
            count = ret.json()["count"] - count_q
            total_count += count

        avg_count = total_count / n_thread

        print(
            f"Time (Completion): {round(time_seconds, 4)}, n threads: {n_thread}, "
            f"throughput: {round(avg_count / time_seconds, 4)} tokens/s, "
        )

        df = pd.DataFrame(
            [
                [
                    args.exp_name,
                    n_thread,
                    round(avg_count / time_seconds, 4),
                    args.num_gpu,
                ]
            ],
            columns=["name", "batch size", "throughput(token/s)", "num gpu"],
        )

        if os.path.exists(test_file_name):
            df.to_csv(test_file_name, mode="a", header=False)
        else:
            df.to_csv(test_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--exp-name", type=str)
    parser.add_argument("--num-gpu", type=str)
    args = parser.parse_args()

    main()
