import socket
import json
import time
HOST = '127.0.0.1'  # 或者使用远程服务器的 IP 地址
PORT = 8888
def run_client(QA_History,cur_question,cur_answer,ground):
    # 创建一个 socket 对象
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 连接到远程服务器
        s.connect((HOST, PORT))
        ####
        #### chatgpt 使用通信解决
        #### 多答案先不解决

        dialogue_content = ''
        role_0 = 'teacher: '
        role_1 = 'student: '
        for item in QA_History[-20:]:
            question_text = item[1]+'\n '
            pred_answer = item[2][0] + '\n '
            dialogue_content += role_0 + question_text + role_1 + pred_answer
        messages = [
            # details in paper
            ]


        ##  Limitation：prompt
        ## metric: 意义


        while True:
            # 我messages这里和api需要的messages格式保持一致，因此传输的时候进行json序列化
            prompt = json.dumps(messages)
            # 编码成二进制，并发送
            print(prompt)
            s.sendall(prompt.encode())
            print('已发送')
            # 从服务器接收消息
            data = s.recv(1024).decode()
            # 打印收到的消息
            print('Received:', repr(data))
            if data!='transmission error':
                break
            time.sleep(0.5)
    return data


if __name__ == '__main__':

    run_client(HOST,PORT)