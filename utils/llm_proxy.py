from whale import TextGeneration
import os
import json
import requests
import time
import numpy as np
TextGeneration.set_api_key("GKCPRES90J", base_url="https://whale-wave.alibaba-inc.com")

flag_public = True
MODEL_NAME='qwen_25_72b'
num_success=0
def output_response(flag, response):
    print("success: ", flag)
    print("response: ", response)
    print(
        "_____________________________________________________________________________________________"
    )


class LLM_Proxy:
    
    def __init__(self):
        self.config = {"max_length": 10000}
    def llm_request_local(self,prompt,port=5378,do_sample=False):
        url=f"http://0.0.0.0:{port}/generate"
        input={"prompt":prompt,"do_sample":do_sample}
        try:
            response = requests.post(url, json=input)  # 将数据以 JSON 格式发送
            # 检查响应状态
            if response.status_code == 200:
                # 请求成功，处理返回结果
                result = response.json()  # 将响应内容解析为 JSON
                result=result[len(prompt):]
                return True,result
            else:
                print("response error==",response)
                # 请求失败，处理错误
                return False, ""

        except requests.exceptions.RequestException as e:
            # 捕获请求过程中可能发生的异常
            print(f"请求过程中发生错误: {e}")
        return False,""


    def llm_request_local_72b(self,prompt,do_sample):
        url = "https://nebula-proxy-gateway.alibaba-inc.com/gateway/domain/mdl_chat/route/0811d69f239a4fef8afa9046fcb4b523/proxy/v1/chat/completions"
        payload = json.dumps({
            "model": "your model name",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "tools": [],
            "do_sample": do_sample,
            "temperature": 0,
            "top_p": 0,
            "n": 1,
            "max_tokens": 1000,
            "stream": False
        })
        headers = {
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': 'nebula-proxy-gateway.alibaba-inc.com',
            'Connection': 'keep-alive'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.text)


    def llm_request_general(self,prompt, model,do_sample,temperature=1, curiosity='lv4'):
        # curiosity : lv1-lv7
        # model : gpt-4-turbo-128k
        print("model==",model)
        if not do_sample:
            data = {
                "messages": [{"role": "user", "content": prompt}],
                "platformInput": {'model': model}, 
                "max_tokens": 800,
                "temperature": 0}
        else:
            data = {
                "messages": [{"role": "user", "content": prompt}],
                "platformInput": {'model': model}, 
                "max_tokens": 800,
                "temperature": temperature}
        
        data = json.dumps(data).encode("UTF-8")
        headers = {
        "X-AK": "8768f9390bd374d4ca6a01b177020701",  # You need to fill in your key
        "Content-Type": "application/json"
        }
        url = "https://idealab.alibaba-inc.com/api/v1/chat/completions"
        response = ""
        try:
            response = requests.post(url=url, data=data, headers=headers)
            response = json.loads(response.content.decode("UTF-8"))
            if response['success']:
                response = response['data']['choices'][0]['message']['content']
                return True, response
            else:
                print(response)
        except BaseException as e:
            print("exception==",e)
            pass
        print("false generation")

        return False, ""


    def request_qwen7b(self,prompt,do_sample):
        url = "https://nebula-proxy-gateway.alibaba-inc.com/gateway/domain/mdl_chat/route/a4e6ecaac456440a98958b9e33c59206/proxy/v1/chat/completions"
        payload = json.dumps({
        "model": "your model name",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "tools": [],
        "do_sample": do_sample,
        "temperature": 1,
        "top_p": 0,
        "n": 1,
        "max_tokens": 0,
        "stream": False
        })
        headers = {
        'Content-Type': 'application/json',
        'Accept': '*/*',
        'Host': 'nebula-proxy-gateway.alibaba-inc.com',
        'Connection': 'keep-alive'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        try:
            print("qwen7b")
            response = response.json()
            #print("response==",response)
            res = response['choices'][0]['message']['content']
            return True, res
        except BaseException as e:
            print("Except",e)
            return False,""


    def request_llama8b(self,prompt,do_sample):
        url = "https://nebula-proxy-gateway.alibaba-inc.com/gateway/domain/mdl_chat/route/2446c5bb92b14e838db7044ff85828c8/proxy/v1/chat/completions"
        payload = json.dumps({
        "model": "your model name",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "tools": [],
        "do_sample": do_sample,
        "temperature": 1,
        "top_p": 0,
        "n": 1,
        "max_tokens": 0,
        "stream": False
        })
        headers = {
        'Content-Type': 'application/json',
        'Accept': '*/*',
        'Host': 'nebula-proxy-gateway.alibaba-inc.com',
        'Connection': 'keep-alive'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        try:
            print("llama8b")
            response = response.json()
            res = response['choices'][0]['message']['content']
            return True, res
        except BaseException as e:
            print("Except",e)
            return False,""

    def request_mistral7b(self,prompt,do_sample):
        url = "https://nebula-proxy-gateway.alibaba-inc.com/gateway/domain/mdl_chat/route/0970015280894de68e3046830ca1211e/proxy/v1/chat/completions"
        payload = json.dumps({
        "model": "your model name",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "tools": [],
        "do_sample": do_sample,
        "temperature": 1,
        "top_p": 0,
        "n": 1,
        "max_tokens": 0,
        "stream": False
        })
        headers = {
        'Content-Type': 'application/json',
        'Accept': '*/*',
        'Host': 'nebula-proxy-gateway.alibaba-inc.com',
        'Connection': 'keep-alive'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        try:
            print("mistral7b")
            response = response.json()
            res = response['choices'][0]['message']['content']
            return True, res
        except BaseException as e:
            print("Except",e)
            return False,""

    def request_deepseek(self,prompt,do_sample=False):
        import os
        from openai import OpenAI

        client = OpenAI(
            api_key = 'd46197c6-ac6c-4277-97a4-c69b9ef4287c',
            base_url = "https://ark.cn-beijing.volces.com/api/v3",
        )
        try:
            # Non-streaming:
            print("----- standard request -----")
            completion = client.chat.completions.create(
                model = "ep-20250214010325-6fkgz",  # your model endpoint ID
                messages = [
                    {"role": "system", "content": "你是deepseek，一个 AI 人工智能助手"},
                    {"role": "user", "content": prompt},
                ],
            )
            print(completion.choices[0].message.content)
            return True,completion.choices[0].message.content
        except:
            return False,""

    def llm_request_qw_25_72b_ours(self, prompt, do_sample, model='qwen_25_72b'):
            url = 'https://alimama-ai-llm-analysis.alibaba-inc.com/api'
            text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" + "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
            gen_config = {"max_new_tokens":5000, "do_sample": do_sample, "beam_width": 1}
            #model_name ="qwen2_72b_instruct"
            #model_name = "qwen2_moe_57B"
            model_name = "qwen2_5_72B_Instruct"

            data = {"model_name":model_name,
                    "mode": "generate",
                    "data":{"text":text, "gen_config":gen_config}}

            r = requests.post(url, json=data, timeout=(6000, 6000))
            response = r.json()
            if "result" in response:
                res = response["result"].split("<|im_start|>assistant\n")[-1]
                return True, res
            else:
                return False, ""

    def llm_request(self, prompt, do_sample=False, model_name="qwen_25_72b",temperature=1,judge=False):
        try:
            return self.llm_request1(prompt,do_sample,model_name,judge=judge,temperature=temperature)
        except BaseException as e:
            import logging
            print("except==",e)
            logging.exception(e)
            return False,""

    def llm_request1(self, prompt, do_sample=False, model_name="qwen_25_72b",temperature=1,judge=False):
        if model_name=='qwen7b':
            return self.request_qwen7b(prompt,do_sample)
        if model_name=='mistral7b':
            return self.request_mistral7b(prompt,do_sample)
        if model_name=='llama8b':
            return self.request_llama8b(prompt,do_sample)
        if model_name=='deepseek':
            return self.request_deepseek(prompt,do_sample)
        if model_name=='qwen2.5-72b-instruct':
            return self.llm_request_qw_25_72b_ours(prompt,do_sample)
        if model_name=='gpt-4o-0513' or model_name=='claude35_haiku' or model_name=='gemini-1.5-pro' or model_name=='llama3' or model_name=='gpt-4o-mini-0718' or model_name=='qwen2.5-72b-instruct' or model_name=='qwen2.5-7b-instruct' or model_name=='claude3_sonnet':
            flag, response=self.llm_request_general(prompt,model=model_name,do_sample=do_sample,temperature=temperature)
            if flag:
                return flag,response
            if not flag:
                for _ in range(5):
                    t=np.random.randint(1,10)
                    time.sleep(t)
                    flag, response=self.llm_request_general(prompt,model=model_name,do_sample=do_sample)
                    if flag:
                        return flag,response
            
            return False,""
        # flag,response=self.llm_request_general(prompt,model=model_name,do_sample=False)
        # if flag==False:
        #     for _ in range(10):
        #         t=np.random.randint(1,4)
        #         time.sleep(t)
        #         flag,response=self.llm_request_general(prompt,model=model_name,do_sample=False)
        #         if flag==True:
        #             break
        # t=np.random.randint(1,4)
        # time.sleep(t)
        # return flag,response

        model_name='qwen_25_72b'
        global MODEL_NAME,num_success
        #model_name='gpt4'
        #return self.llm_request_local_72b(prompt,do_sample)
        if model_name:
            llm_name = model_name
        else:
            llm_name = os.environ.get("llm_model_name", "qwen72b")
        if llm_name == "qwen72b":
            result= self.llm_request_whale_72b(prompt,do_sample=do_sample)
        elif llm_name == "qwen_25_72b":
            result= self.llm_request_qw_25_72b(prompt,do_sample=do_sample)
        elif llm_name == "qwen_a14b":
            result= self.llm_request_qwen_a14b(prompt,do_sample=do_sample)
        elif llm_name == "gpt4":
            result= self.llm_request_gpt4(prompt,do_sample=do_sample)
        else:
            result=self.llm_request_local(prompt,port=model_name,do_sample=do_sample)
            return result
        if result[0]==False:
            print("72b error try a14b and gpt")
            try:
                return self.llm_request_qwen_a14b(prompt,do_sample=do_sample)
            except:
                return self.llm_request_general(prompt,model='gpt-4o-mini-0718',do_sample=do_sample)
        
        if result[1]=="error":
            print("error generated")
            for _ in range(10):
                result=self.llm_request_qw_25_72b(prompt,do_sample=do_sample)
                if result[1]!="error":
                    return result
            return False,""
        return result
        #raise RuntimeError("Invalid LLM.")


    def llm_request_whale_72b(self, inp_ori,do_sample=False):
        print("whale 72b")
        temperature=1 if do_sample else 0
        config = {
            "max_length": 1000,
            "do_sample": do_sample,
            "topk": 0.1,
            "temperature": temperature,
        }
        inp = "<|im_start|>human\n{}<|im_start|>assistant\n".format(inp_ori)
        response = TextGeneration.call(
            model="Qwen-72B-Chat-Pro",
            prompt=inp,
            timeout=5000,
            streaming=False,
            generate_config=config,
        )
        # 处理结果
        if response.status_code == 200:
            # print(response.output['response'])
            return True, response.output["response"]
        else:
            for _ in range(5):
                t=np.random.randint(1,10)
                time.sleep(t)
                response = TextGeneration.call(
                    model="Qwen-72B-Chat-Pro",
                    prompt=inp,
                    timeout=5000,
                    streaming=False,
                    generate_config=config,
                )
                if response.status_code == 200:
                    # print(response.output['response'])
                    return True, response.output["response"]
                msg = "error_code: [%d], error_message: [%s]" % (
                    response.status_code,
                    response.status_message,
                )
                print("error happen ",msg)
            return False, ""

    def llm_request_qw72b(self, prompt, model="qwen_72b"):
        url = "http://alimama-scs-llm-textgeneration.alibaba-inc.com/api"
        text = "<|im_start|>human\n" + prompt + "<|im_start|>assistant\n"
        text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            + "<|im_start|>user\n"
            + prompt
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        gen_config = {
            "max_length": 10000,
            "do_sample": True,
            "top_p": 0.7,
            "temperature": 0.7,
        }
        model_name = "qwen_72b"

        data = {
            "model_name": model_name,
            "mode": "generate",
            "data": {"text": text, "gen_config": gen_config},
        }

        r = requests.post(url, json=data, timeout=(500, 500))
        response = r.json()
        if "result" in response:
            res = response["result"].split("<|im_start|>assistant\n")[-1]
            return True, res
        else:
            return False, ""

    def llm_request_qwen_a14b(self, prompt,do_sample=False, model="qwen_a14b"):
        print("request a 14b")
        url = "http://alimama-aigc-service.alibaba-inc.com/api"
        text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            + "<|im_start|>user\n"
            + prompt
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        temperature=1 if do_sample else 0
        gen_config = {
            "max_length": 1000,
            "do_sample": do_sample,
            "topk": 0.1,
            "temperature": temperature,
             "beam_width": 1
        }
        model_name = "qwen2_moe_57B"

        data = {
            "model_name": model_name,
            "mode": "generate",
            "data": {"text": text, "gen_config": gen_config},
        }
        r = requests.post(url, json=data, timeout=(1000, 1000))
        response = r.json()
        if "result" in response:
            res = response["result"].split("<|im_start|>assistant\n")[-1]
            return True, res
        else:
            return False, ""

    def llm_request_ds(self, prompt, do_sample=True, model="ds_r1"):
        url = 'http://alimama-deepseek.vipserver/api'
        url = 'http://alimama-deepseek.alibaba-inc.com/api'
        text = prompt

        # 暂时只支持这两个，其他参数陆续支持中
        gen_config = {"max_tokens": 2048,
            "temperature": 1
            }
        import uuid
        model_name = "deepseek_v3_671b"
        unique_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f'{time.time()}'))
        data = {"model_name": model_name, # 必填：deepseek模型名称
                "source":"xiaowan",          # 必填：请求方标识，填充xiaowan
                "request_id":unique_id,     # 选填：请求唯一标识，便于定位
                "mode": "chat",           # 必填：暂时只支持chat
                "data":{"text":"",
                        "messages":[
                            {
                                "content": "You are a helpful and harmless assistant. You should think step-by-step.",
                                "role": "system"
                            },
                            {
                                "content": text,
                                "role": "user"
                            },
                            ], 
                        "gen_config":gen_config}}

        r = requests.post(url, json=data, timeout=(5000, 5000))
        response = r.json()
        if "result" in response:
            res = response["result"]
            return True, res
        else:
            return False, ""

    def llm_request_qw_25_72b(self, prompt, do_sample=False, model="qwen_25_72b"):
        print("qwen 72b")
        url = "http://alimama-aigc-service.alibaba-inc.com/api"
        text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            + "<|im_start|>user\n"
            + prompt
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        temperature=1.3 if do_sample else 0
        gen_config = {"max_new_tokens": 2000, "do_sample": do_sample, "beam_width": 1, "temperature":temperature}
        # model_name ="qwen2_72b_instruct"
        # model_name = "qwen2_moe_57B"
        model_name = "qwen2_5_72B_Instruct"

        data = {
            "model_name": model_name,
            "mode": "generate",
            "data": {"text": text, "gen_config": gen_config},
        }

        r = requests.post(url, json=data, timeout=(1000, 1000))
        try:
            response = r.json()
        except BaseException as e:
            print("exception qwen==",str(e))
            print("response==",r)
            return False,""
        if "result" in response:
            res = response["result"].split("<|im_start|>assistant\n")[-1]
            return True, res
        else:
            print("res==",response)
            return False, ""

    def llm_request_gpt4(self, prompt, model="gpt-4-turbo-128k", curiosity="lv4", do_sample=False):
        # curiosity : lv1-lv7
        # model : gpt-4-turbo-128k


        data = {"model": model, "prompt": prompt, "curiosity": curiosity}
        data["messages"] = [{"role": "user", "content": prompt}]
        data = json.dumps(data).encode("UTF-8")
        headers = {
            "X-AK": "6690de90d61d8a15956fb2ddadb6e439",  # You need to fill in your key
            "Content-Type": "application/json",
            "Authorization": f"Bearer 6690de90d61d8a15956fb2ddadb6e439",
        }

        url = "https://idealab.alibaba-inc.com/aigc/v1/askTextToTextMsg"
        url = "https://idealab.alibaba-inc.com/api/openai/v1/chat/completions"
        try:
            response = requests.post(url=url, data=data, headers=headers)
            response = json.loads(response.content.decode("UTF-8"))
            print("response==",response)
            with open('test.json','w') as f:
                json.dump(response,f,indent=2,ensure_ascii=False)
            response = response["choices"][0]["message"]["content"]

            return True, response
        except BaseException as e:
            print("exception==",str(e))
            return False, ""


if __name__ == "__main__":
    llm_proxy = LLM_Proxy()
    import time
    #inp_ori = "st nam     Caused by: com.ibm.mq.MQException: JMSCMQ0001: WebSphere MQ call failed\n     with compcode '2' ('MQCC_FAILED') reason '2035' ('MQRC_NOT_AUTHORIZED').\n      at\n     com.ibm.msg.client.wmq.common.internal.Reason.createException(Reason.jav\n     a:204)\n      ... 162 more\n\nThe same MQ service is working fine in the old BPM environment. User name and password configured for the queue is correct. However, when we check the MQ logs, we observing the username is not used, instead the sys admin user seems to be used."
    start=time.time()
    model='qwen2.5-7b-instruct'
    for _ in range(10):
        inp_ori="who is newton, response within 10 tokens"
        #flag, response = llm_proxy.request_qwen7b(inp_ori,do_sample=False)
        flag,response=llm_proxy.llm_request_general(inp_ori,model='gpt-4o-mini-0718',do_sample=True)
        #flag,response=llm_proxy.llm_request(inp_ori,model_name='qwen72b',do_sample=True)
        #flag,response=llm_proxy.llm_request(inp_ori,do_sample=True)
        #flag,response=llm_proxy.llm_request_whale_72b(inp_ori)
        #flag,response=llm_proxy.llm_request_ds(inp_ori)
        
        #flag,response=llm_proxy.llm_request_qwen_a14b(inp_ori)
        print(flag,response)
    end=time.time()
    print("model time=",end-start)




