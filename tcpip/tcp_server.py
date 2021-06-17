import socket

tcp_client_socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

server_ip='192.168.1.9'
server_sport=9090

tcp_client_socket.connect((server_ip,server_sport))

send_data=input("请输⼊要发送的数据：")

tcp_client_socket.send(send_data.encode("gbk"))

recvData=tcp_client_socket.recv(1024)

print('接收到的数据为:', recvData.decode('gbk'))

tcp_client_socket.close()

