ps -eo user,pid,cmd|grep fastchat.serve.model_worker|grep -v grep|awk '{print $2}'|xargs kill -9
ps -eo user,pid,cmd|grep fastchat.serve.vllm_worker|grep -v grep|awk '{print $2}'|xargs kill -9